"""Regression tests for row construction during ingest.

The bug these pin down: `_rows_for` unpacked three names from
`zip(chunks, vectors)`, which yields 2-tuples. Every write therefore raised
`ValueError: not enough values to unpack (expected 3, got 2)`.

It went unnoticed because `ingest_chunks` returns early ("unchanged") when the
source hash matches, so re-ingesting an existing page never reached the buggy
line. Only NEW content failed, and the caller saw it as an HTTP 413.

These tests deliberately avoid LanceDB and the embedding model.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag import store


def _chunks(*specs):
    return [Document(page_content=text, metadata={"page": page}) for text, page in specs]


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
