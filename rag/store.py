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
# Kept so the table can be reopened cheaply. `_table` is a snapshot and does not
# refresh when another handle commits; see `fresh_table`. Holding the connection
# avoids reaching for the private `table._conn`.
_db = None

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

    global _db
    _db = db
    _table = table
    logger.info("LanceDB ready at %s (dim=%d)", db_path or config.DB_PATH, dim)
    return table


# ---------------------------------------------------------------------------
# DEPRECATED: cached-handle accessors
# ---------------------------------------------------------------------------
# `get_table` and `try_get_table` return the process-lifetime handle `_table`,
# which is a LanceDB snapshot opened at startup. It does not observe commits made
# through any other handle, so after an external delete or a table rebuilt on
# disk it keeps answering from the version it was opened at. That was issue #22:
# `ingest` reported `unchanged` for content that was not stored, and `search`
# served rows that no longer existed.
#
# Nothing in the application calls these any more -- ingest, search and /health
# all use `fresh_table` / `try_fresh_table`. They are kept only because they are
# part of the module's surface and something outside this repository (a script,
# an ad-hoc `python -c`) may import them.
#
# Do not add new call sites. Use `fresh_table()` when the table must exist and
# `try_fresh_table()` when a cold start is a legitimate state.


def get_table():
    """DEPRECATED: returns the stale cached handle. Use `fresh_table()`."""
    if _table is None:
        raise RuntimeError("Table is not initialized")
    return _table


def try_get_table():
    """DEPRECATED: returns the stale cached handle. Use `try_fresh_table()`.

    Retained for callers that must not raise on a cold start. Any caller that
    reads through the returned handle risks seeing a pre-commit snapshot.
    """
    return _table


def fresh_table():
    """Reopen the table so the returned handle sees the current version.

    A LanceDB table handle is a snapshot. `_table` is opened once at startup and
    never refreshed, so after any commit made through another handle -- an
    external delete, a second writer, a table rebuilt on disk -- the cached
    handle keeps answering from the version it was opened at. Measured:

        held handle after external delete : 1 row
        reopened handle                   : 0 rows
        held handle again                 : 1 row

    That is issue #22: `ingest` answered `unchanged` for content that was not
    stored, and `search` served rows that no longer existed.

    Reopening is cheap next to embedding, which dominates an ingest by orders
    of magnitude, and writes are already serialised per source by `_lock_for`.
    """
    if _table is None or _db is None:
        raise RuntimeError("Table is not initialized")
    return _db.open_table("documents")


def try_fresh_table():
    """`fresh_table`, or None before initialisation."""
    if _table is None or _db is None:
        return None
    return _db.open_table("documents")


def reset_table() -> None:
    global _table, _db
    _table = None
    _db = None


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def escape(value: str) -> str:
    """Escape a value for LanceDB's SQL-like literal syntax."""
    return value.replace("'", "''")


def delete_existing_source(table, source: str) -> None:
    table.delete(f"source = '{escape(source)}'")


def delete_source(source: str, *, replace_also: Iterable[str] = ()) -> tuple[int, list[str]]:
    """Delete every stored row for a source, across its alias keys.

    Returns ``(rows_deleted, keys_tried)``.

    Two properties matter, and neither is optional:

    * **Aliases are deleted too.** A document may sit under several keys --
      ``url:`` and the legacy unprefixed form, the bare and slashed spellings of
      a root URL -- because that is what `api.source_aliases()` exists for. A
      delete that removed only the primary key would leave the others behind and
      the content would still be retrievable, which is the one outcome a delete
      must not have.

    * **The same per-source lock as ingest.** Deleting outside it races
      `ingest_chunks`: a write that checked for an existing generation just
      before the delete would add rows for a source the caller was told no
      longer exists. Reusing `_lock_for` serialises the two without introducing
      a second lock order to reason about.

    A source that does not exist is not an error: `rows_deleted` is 0 and the
    caller can say so. Deleting is idempotent by nature.
    """
    keys = [k for k in dict.fromkeys((source, *replace_also)) if k]
    total = 0
    # Fresh handle: a snapshot opened earlier may not see the latest commits, so
    # the count afterwards would be read from a stale view and could report a
    # successful delete over rows that are still there (issue #22).
    with _lock_for(source):
        table = fresh_table()
        for key in keys:
            before = count_rows_for(key, table)
            if before:
                delete_existing_source(table, key)
                total += before

        # Verify against a reopened handle rather than trusting `delete`. Same
        # reasoning as the post-write check in `ingest_chunks`: a call returning
        # without raising is not evidence that the rows are gone.
        leftover = count_rows_for(source, fresh_table())
        if leftover:
            raise RuntimeError(
                f"Delete of {source} reported success but {leftover} rows remain. "
                "The delete did not land; refusing to report success."
            )

    return total, keys


def count_rows_for(source: str, table=None) -> int:
    """Number of stored rows for `source`, read through `table`.

    Used to verify a write actually landed. Opens a fresh handle by default, so
    the count is not taken from a snapshot that might predate the write.
    """
    handle = table if table is not None else fresh_table()
    return len(
        handle.search()
        .where(f"source = '{escape(source)}'")
        .select(["source"])
        .limit(100000)
        .to_list()
    )


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

    # A fresh handle, not the process-lifetime one: a snapshot opened at startup
    # does not observe commits made elsewhere, which made `unchanged` a lie and
    # kept deleted rows in search results. See `fresh_table`.
    table = fresh_table()
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

        # Do not take the write on trust. `table.add` returning without raising
        # is not evidence that the rows are visible, and issue #22 was precisely
        # a confident `ingested` over an empty table. Re-read through a newly
        # opened handle and fail loudly rather than reporting success.
        written = count_rows_for(source, fresh_table())
        if written != len(rows):
            raise RuntimeError(
                f"Ingest of {source} reported {len(rows)} chunks but the table "
                f"holds {written} afterwards. The write did not land; refusing "
                f"to report success. This usually means the table on disk was "
                f"replaced underneath a running process -- restart the service."
            )

    return len(rows), "ingested"
