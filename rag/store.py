"""LanceDB access: embedding, writing and reading chunks.

Two correctness fixes live here relative to the original single-file version:

* **Atomic ingest.** The old code deleted a source's rows and then added the
  new ones. If the add failed, the source's data was gone with no way back.
  `ingest_chunks()` now adds the new rows before deleting the old ones, both
  inside the write lock. A failure at any point leaves the previous rows
  intact.

* **Explicit hash lookup errors.** The old `existing_source_hash()` swallowed
  every exception and returned `None`, which means "not stored yet" — so any
  query fault caused every file to be silently re-ingested on every run.
  Query faults now propagate; only a genuine miss returns `None`.
"""

from __future__ import annotations

import hashlib
import logging
import threading

import lancedb
import numpy as np
import pyarrow as pa
from langchain_core.documents import Document

from . import config

logger = logging.getLogger("rag.store")

_embeddings = None
_table = None
_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def init_embeddings():
    """Load the embedding model. Called once from the lifespan handler."""
    global _embeddings
    from langchain_huggingface import HuggingFaceEmbeddings

    _embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
    return len(_embeddings.embed_query("dimension check"))


def get_embeddings():
    if _embeddings is None:
        raise RuntimeError("Embedding model is not initialized")
    return _embeddings


def reset_embeddings() -> None:
    global _embeddings
    _embeddings = None


def embed_in_batches(texts: list[str]) -> list[list[float]]:
    model = get_embeddings()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), config.EMBED_BATCH_SIZE):
        vectors.extend(model.embed_documents(texts[i : i + config.EMBED_BATCH_SIZE]))
    return vectors


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


def _schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("vector", lancedb.vector(dim)),
            pa.field("source", pa.string()),
            pa.field("page", pa.int64()),
            pa.field("chunk_id", pa.int64()),
            pa.field("text", pa.string()),
            pa.field("file_hash", pa.string()),
        ]
    )


def init_table(db_path: str | None = None):
    """Create or open the documents table. Returns the table."""
    global _table
    dim = len(get_embeddings().embed_query("dimension check"))
    db = lancedb.connect(db_path or config.DB_PATH)
    table = db.create_table("documents", schema=_schema(dim), exist_ok=True)

    try:
        existing_dim = table.schema.field("vector").type.list_size
    except Exception:
        existing_dim = None
    if existing_dim is not None and existing_dim != dim:
        raise RuntimeError(
            f"Existing table vector dimension is {existing_dim}, but the "
            f"configured model produces {dim}. Use a new database path or "
            "rebuild the table."
        )

    _table = table
    logger.info("LanceDB ready at %s (dim=%d)", db_path or config.DB_PATH, dim)
    return table


def get_table():
    if _table is None:
        raise RuntimeError("Table is not initialized")
    return _table


def reset_table() -> None:
    global _table
    _table = None


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def escape(value: str) -> str:
    """Escape a value for LanceDB's SQL-like literal syntax."""
    return value.replace("'", "''")


def delete_existing_source(table, source: str) -> None:
    table.delete(f"source = '{escape(source)}'")


def existing_source_hash(table, source: str) -> str | None:
    """Return the stored file_hash for a source, or None if not stored.

    A query fault propagates. Returning None on error would be indistinguishable
    from "not stored yet" and would force a full re-ingest on every call.
    """
    rows = (
        table.search()
        .where(f"source = '{escape(source)}'")
        .select(["file_hash"])
        .limit(1)
        .to_list()
    )
    return rows[0]["file_hash"] if rows else None


def calculate_file_hash(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def _rows_for(
    chunks: list[Document], source: str, file_hash: str, vectors: list[list[float]]
) -> list[dict]:
    return [
        {
            "vector": np.asarray(vec, dtype=np.float32).tolist(),
            "source": source,
            "page": int(chunk.metadata.get("page", 0) or 0),
            "chunk_id": i,
            "text": chunk.page_content,
            "file_hash": file_hash,
        }
        # zip() over two iterables yields 2-tuples. This previously unpacked
        # three names, which raised ValueError on EVERY write - silently, since
        # an unchanged source short-circuits before reaching here.
        for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True))
    ]


def ingest_chunks(
    chunks: list[Document],
    source: str,
    file_hash: str,
    *,
    max_chunks: int | None = None,
) -> tuple[int, str]:
    """Embed and store chunks for one source.

    Returns (chunks_written, outcome), outcome being
    ``ingested`` | ``unchanged`` | ``empty``.

    Raises ``ValueError`` if the chunk count would exceed ``max_chunks``.

    Ordering matters: new rows are added *before* the old ones are deleted, so
    an add failure leaves the previously stored content intact.
    """
    table = get_table()

    if existing_source_hash(table, source) == file_hash:
        return 0, "unchanged"

    kept = [c for c in chunks if c.page_content.strip()]
    if not kept:
        return 0, "empty"

    if max_chunks is not None and len(kept) > max_chunks:
        raise ValueError(
            f"{len(kept)} chunks for {source} exceeds the per-request limit of {max_chunks}."
        )

    texts = [c.page_content for c in kept]
    vectors = embed_in_batches(texts)
    rows = _rows_for(kept, source, file_hash, vectors)

    with _write_lock:
        table.add(rows)
        # Only after the new rows are durable do we drop the old generation.
        table.delete(f"source = '{escape(source)}' AND file_hash != '{escape(file_hash)}'")

    return len(rows), "ingested"
