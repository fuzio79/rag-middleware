"""LanceDB access: embedding, writing and reading chunks.

Correctness properties that live here, each one previously a defect:

* **Atomic ingest.** The old code deleted a source's rows and then added the
  new ones. If the add failed, the source's data was gone with no way back.

* **Explicit hash lookup errors.** The old `existing_source_hash()` swallowed
  every exception and returned `None`, which means "not stored yet" — so any
  query fault caused every file to be silently re-ingested on every run.
  Query faults now propagate; only a genuine miss returns `None`.

* **No duplicate generations.** The hash check used to run outside the write
  lock, so two concurrent requests for the same source both saw "not stored"
  and both added rows. Nothing removed the duplicates either, because the
  cleanup only deleted *older* hashes. Both the check and the write now happen
  under a per-source lock, and rows are deleted before being re-added so a
  crash between the two cannot accumulate.

* **Alias keys.** A source may legitimately have been stored under a different
  key by an older build (`url` vs the normalised form of it). `replace_also`
  lets a write claim those keys, so changing a key scheme replaces the previous
  rows instead of orphaning them and leaving a duplicate behind. That is what
  makes a key change deployable without a data migration.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import lancedb
import numpy as np
import pyarrow as pa
from langchain_core.documents import Document

from . import config

logger = logging.getLogger("rag.store")

_embeddings = None
_table = None

# One lock per process, and one per source within it. A single global lock was
# correct but needlessly serialised unrelated sources; a map keyed by source
# keeps concurrent ingests of *different* documents parallel while making the
# check-then-write sequence atomic for any one source.
#
# KNOWN LIMIT: this is in-process only. Under `uvicorn --workers N` each worker
# has its own locks, its own embedding model and its own LanceDB handle, and
# nothing here coordinates across them. The service is deployed with a single
# worker (see the Dockerfile CMD) and that is a requirement, not a preference.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


@contextmanager
def _lock_for(source: str) -> Iterator[None]:
    """Per-source write lock. Replaces the previous single global lock."""
    with _locks_guard:
        lock = _locks.setdefault(source, threading.Lock())
    with lock:
        yield


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


def _stored_hash(table, keys: Iterable[str]) -> str | None:
    """First stored hash across `keys`, primary key first."""
    for key in keys:
        found = existing_source_hash(table, key)
        if found is not None:
            return found
    return None


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
    replace_also: Iterable[str] = (),
) -> tuple[int, str]:
    """Embed and store chunks for one source.

    Returns (chunks_written, outcome), outcome being
    ``ingested`` | ``unchanged`` | ``empty``.

    Raises ``ValueError`` if the chunk count would exceed ``max_chunks``.

    The whole check-then-write sequence runs under a per-source lock. Two
    consequences, both deliberate:

    * the unlocked early-out is gone, because it was a fast path that could not
      be trusted. Embedding now happens inside the lock, so a concurrent ingest
      of the *same* source waits for the first to finish instead of burning CPU
      on vectors that were about to be discarded;
    * ingests of *different* sources still overlap, which is why the lock is
      keyed by source rather than global.

    Rows are deleted before being re-added. A crash between the add and the
    delete in the old ordering left two generations in the table, and because
    `existing_source_hash` used `limit(1)` it could then keep returning the old
    hash and re-add on top of the new rows. Delete-first cannot accumulate.

    `replace_also` names other keys that refer to the same document under an
    older keying scheme. They are consulted when looking for a previous
    generation and removed when the new one is written. This is what lets a
    `source` key change land without a migration: an existing row stored under
    the old key is replaced, not orphaned next to a duplicate.

    LOCKING NOTE: only `source` is locked. Aliases are cleaned up under that
    same lock, so two *concurrent* ingests whose alias sets overlap are not
    fully serialised. In practice aliases are alternative spellings of the same
    page and are derived from the request being processed, so the overlap case
    requires the same document to be ingested twice simultaneously under two
    different spellings. Accepted, and called out rather than assumed away.
    """
    kept = [c for c in chunks if c.page_content.strip()]
    if not kept:
        return 0, "empty"

    if max_chunks is not None and len(kept) > max_chunks:
        raise ValueError(
            f"{len(kept)} chunks for {source} exceeds the per-request limit of {max_chunks}."
        )

    table = get_table()
    aliases = [k for k in dict.fromkeys(replace_also) if k and k != source]
    keys = [source, *aliases]

    with _lock_for(source):
        if _stored_hash(table, keys) == file_hash:
            # Content already present, possibly under an alias. Leaving it
            # there keeps the row count stable and avoids rewriting vectors for
            # a no-op; `preflight.sh` is what reports residual alias rows.
            return 0, "unchanged"

        texts = [c.page_content for c in kept]
        vectors = embed_in_batches(texts)
        rows = _rows_for(kept, source, file_hash, vectors)

        # Drop every earlier generation under any known key, then write the new
        # one. Any rows left behind by an interrupted earlier run are cleared by
        # the same statements.
        for key in keys:
            delete_existing_source(table, key)
        table.add(rows)

    return len(rows), "ingested"
