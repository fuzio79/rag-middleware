"""Regression tests for row construction and write ordering during ingest.

The first bug these pin down: `_rows_for` unpacked three names from
`zip(chunks, vectors)`, which yields 2-tuples. Every write therefore raised
`ValueError: not enough values to unpack (expected 3, got 2)`.

It went unnoticed because `ingest_chunks` returns early ("unchanged") when the
source hash matches, so re-ingesting an existing page never reached the buggy
line. Only NEW content failed, and the caller saw it as an HTTP 413.

The second: the hash check ran outside the write lock, so two concurrent
ingests of one source both saw "not stored" and both added rows, and nothing
removed the duplicates because the cleanup only deleted *older* hashes.

These tests deliberately avoid the embedding model. LanceDB is faked with a
minimal in-memory table that records the order of `delete` and `add`, because
that ordering is the property being asserted.
"""

from __future__ import annotations

import threading

import pytest
from langchain_core.documents import Document

from rag import store


def _chunks(*specs):
    return [Document(page_content=text, metadata={"page": page}) for text, page in specs]


# ---------------------------------------------------------------------------
# row construction
# ---------------------------------------------------------------------------


def test_rows_pair_chunks_with_vectors():
    chunks = _chunks(("alpha", 1), ("beta", 2))
    rows = store._rows_for(chunks, "src", "hash", [[0.1, 0.2], [0.3, 0.4]])

    assert [r["text"] for r in rows] == ["alpha", "beta"]
    assert [r["page"] for r in rows] == [1, 2]
    assert [r["chunk_id"] for r in rows] == [0, 1]
    assert all(r["source"] == "src" and r["file_hash"] == "hash" for r in rows)


def test_row_text_comes_from_chunk_content():
    rows = store._rows_for(_chunks(("only", 0)), "s", "h", [[1.0]])
    assert rows[0]["text"] == "only"


def test_missing_page_metadata_defaults_to_zero():
    rows = store._rows_for([Document(page_content="no page", metadata={})], "s", "h", [[1.0]])
    assert rows[0]["page"] == 0


def test_mismatched_lengths_raise():
    """`strict=True` must still catch a real chunk/vector mismatch."""
    with pytest.raises(ValueError):
        store._rows_for(_chunks(("a", 0), ("b", 1)), "s", "h", [[0.1]])


# ---------------------------------------------------------------------------
# write ordering and locking
# ---------------------------------------------------------------------------


class _FakeTable:
    """Records the sequence of mutations. Enough to assert ordering."""

    def __init__(self, existing: dict[str, int] | None = None):
        self.ops: list[str] = []
        self.rows: dict[str, int] = existing or {}

    def delete(self, predicate: str):
        self.ops.append("delete")
        self.rows.clear()

    def add(self, rows):
        self.ops.append("add")
        self.rows["n"] = len(rows)


@pytest.fixture
def fake_store(monkeypatch):
    """Wire `ingest_chunks` to a fake table and a fake embedder."""
    table = _FakeTable()

    monkeypatch.setattr(store, "get_table", lambda: table)
    monkeypatch.setattr(store, "embed_in_batches", lambda texts: [[0.0] for _ in texts])
    state = {"hash": None}

    monkeypatch.setattr(store, "existing_source_hash", lambda t, s: state["hash"])

    def set_hash(value):
        state["hash"] = value

    return table, set_hash


def test_existing_generation_is_deleted_before_new_rows_are_added(fake_store):
    """Delete-first, not add-then-delete.

    Add-then-delete left two generations behind when the process died between
    the two calls, and `limit(1)` could then keep returning the stale hash.
    """
    table, set_hash = fake_store
    set_hash("old-hash")

    count, outcome = store.ingest_chunks(_chunks(("a", 0)), "src", "new-hash")

    assert (count, outcome) == (1, "ingested")
    assert table.ops == ["delete", "add"]


def test_unchanged_source_does_not_write(fake_store):
    table, set_hash = fake_store
    set_hash("same-hash")

    count, outcome = store.ingest_chunks(_chunks(("a", 0)), "src", "same-hash")

    assert (count, outcome) == (0, "unchanged")
    assert table.ops == []


def test_empty_chunks_write_nothing(fake_store):
    table, set_hash = fake_store
    count, outcome = store.ingest_chunks(_chunks(("   ", 0)), "src", "h")
    assert (count, outcome) == (0, "empty")
    assert table.ops == []


def test_same_source_is_serialised(fake_store):
    """Two threads ingesting one source must not interleave check and write.

    Before the lock covered the hash check, both threads saw "not stored" and
    both wrote. The fake records every mutation, so an interleaving shows up as
    two adds arriving without their deletes between them.
    """
    table, set_hash = fake_store
    set_hash(None)

    real_embed = store.embed_in_batches

    def slow_embed(texts):
        # Widen the window between the hash check and the write.
        threading.Event().wait(0.02)
        return real_embed(texts)

    store.embed_in_batches = slow_embed
    try:
        threads = [
            threading.Thread(target=store.ingest_chunks, args=(_chunks(("a", 0)), "src", "h1")),
            threading.Thread(target=store.ingest_chunks, args=(_chunks(("b", 0)), "src", "h2")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        store.embed_in_batches = real_embed

    # Whatever the winner, delete/add must arrive as contiguous pairs:
    # never delete, delete, add, add.
    assert table.ops.count("add") == 2
    for i in range(0, len(table.ops) - 1, 2):
        assert table.ops[i : i + 2] == ["delete", "add"], table.ops


def test_different_sources_are_not_serialised(fake_store):
    """Per-source locking, not one global lock.

    The barrier makes the property observable: both threads must be inside
    `embed_in_batches` at the same time. With a single global lock the second
    thread can never get there, the barrier times out, and this fails instead
    of hanging.
    """
    _table, set_hash = fake_store
    set_hash(None)

    barrier = threading.Barrier(2, timeout=5)
    real_embed = store.embed_in_batches
    errors: list[BaseException] = []

    def barrier_embed(texts):
        try:
            barrier.wait()
        except threading.BrokenBarrierError as exc:  # pragma: no cover - regression path
            errors.append(exc)
            raise
        return real_embed(texts)

    store.embed_in_batches = barrier_embed

    def run(source: str) -> None:
        try:
            store.ingest_chunks(_chunks(("x", 0)), source, "h")
        except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(name,)) for name in ("src-a", "src-b")]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        store.embed_in_batches = real_embed

    assert errors == [], f"ingests of different sources blocked each other: {errors}"


def test_lock_for_yields_fresh_managers_and_releases():
    """`_lock_for` yields a fresh context manager and releases on exit.

    It is deliberately NOT reentrant: nesting two acquisitions of the same
    source deadlocks on the underlying non-reentrant `threading.Lock`, which is
    why `ingest_chunks` acquires it exactly once. Asserting that here keeps
    someone from nesting it and shipping a hang, and the sequential re-acquire
    proves the lock is actually released between ingests.
    """
    first = store._lock_for("a")
    second = store._lock_for("a")
    assert first is not second

    with first:
        pass
    with second:  # hangs if the previous acquisition was not released
        pass
