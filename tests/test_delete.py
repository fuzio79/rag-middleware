"""Tests for `POST /api/v1/rag/delete`.

The endpoint exists because nothing could remove a document: the only option was
`clean.sh`, which drops the whole table. That makes the failure modes here
unusually costly, so the tests concentrate on the two things a naive delete gets
wrong:

* **Aliases.** A document can sit under several keys (`url:` vs the legacy raw
  URL, the bare and slashed spellings of a root URL). A delete that misses one
  leaves the content answerable by search while reporting success.
* **Verification.** `table.delete()` returning without raising is not evidence
  the rows are gone. The store re-reads and raises if any remain; the route must
  surface that as a failure rather than a success.

Driven against an in-memory LanceDB table, not the live one.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from rag import api, store


@pytest.fixture
def temp_table(monkeypatch):
    """A throwaway LanceDB table with embedding stubbed to a constant vector.

    These are row-lifecycle tests. Loading the real model would add seconds per
    test and prove nothing extra about deletion.

    The stub must be installed, not merely reset: an earlier version set
    `_embeddings = None` and then called `init_table()`, which needs a real
    model to learn the vector dimension -- every test errored on
    "Embedding model is not initialized".
    """

    class _StubEmbeddings:
        dim = 8

        def embed_query(self, _text):
            return [0.1] * self.dim

        def embed_documents(self, texts):
            return [[0.1] * self.dim for _ in texts]

    tmp = tempfile.mkdtemp()
    monkeypatch.setattr(store, "_embeddings", _StubEmbeddings())
    monkeypatch.setattr(store.config, "DB_PATH", str(Path(tmp) / "db"))
    store.init_table()
    yield
    store.reset_table()
    store.reset_embeddings()


def _add(source: str, n: int = 2, text: str = "body") -> None:
    from langchain_core.documents import Document

    docs = [Document(page_content=f"{text} {i}", metadata={"page": 0}) for i in range(n)]
    store.ingest_chunks(docs, source, f"hash-{source}")


def _delete(source: str, **kwargs):
    return asyncio.run(api.delete_source(api.DeleteRequest(source=source, **kwargs)))


# ---------------------------------------------------------------------------
# candidate generation
# ---------------------------------------------------------------------------


def test_a_raw_url_claims_the_prefixed_and_legacy_keys():
    """The caller should not need to know which spelling the build wrote."""
    candidates = api._delete_candidates("https://example.com/a")
    assert "https://example.com/a" in candidates
    assert "url:https://example.com/a" in candidates


def test_a_prefixed_url_is_also_accepted():
    candidates = api._delete_candidates("url:https://example.com/a")
    assert "url:https://example.com/a" in candidates
    assert "https://example.com/a" in candidates


def test_an_upload_claims_both_filename_forms():
    candidates = api._delete_candidates("notes.txt")
    assert "notes.txt" in candidates
    assert "upload:notes.txt" in candidates


def test_a_prefixed_upload_is_unwrapped():
    candidates = api._delete_candidates("upload:notes.txt")
    assert "upload:notes.txt" in candidates
    assert "notes.txt" in candidates


def test_root_slash_spellings_are_both_claimed():
    """A bare authority has two spellings, and both may be stored."""
    candidates = api._delete_candidates("https://example.com")
    assert "url:https://example.com" in candidates
    assert "url:https://example.com/" in candidates


def test_candidates_are_deduplicated():
    candidates = api._delete_candidates("https://example.com/a")
    assert len(candidates) == len(set(candidates))


def test_a_malformed_url_does_not_raise():
    """`_delete_candidates` must be total; callers pass model-supplied text."""
    candidates = api._delete_candidates("http://[abc")
    assert candidates  # something is always returned


def test_an_empty_source_is_rejected_by_the_model():
    """A blank source must not reach the store: it would be an empty key."""
    with pytest.raises(ValidationError):
        api.DeleteRequest(source="")


# ---------------------------------------------------------------------------
# actual deletion
# ---------------------------------------------------------------------------


def test_deleting_removes_the_rows(temp_table):
    _add("url:https://example.com/a", 3)
    assert store.count_rows_for("url:https://example.com/a") == 3

    result = _delete("https://example.com/a")

    assert result["status"] == "success"
    assert result["chunks_deleted"] == 3
    assert store.count_rows_for("url:https://example.com/a") == 0


def test_deleting_a_url_also_removes_the_legacy_unprefixed_rows(temp_table):
    """The aliasing that matters: a pre-#13 row must not survive the delete."""
    _add("https://example.com/a", 2)  # legacy spelling
    _add("url:https://example.com/a", 2)  # current spelling

    result = _delete("https://example.com/a")

    assert result["chunks_deleted"] == 4
    assert store.count_rows_for("https://example.com/a") == 0
    assert store.count_rows_for("url:https://example.com/a") == 0


def test_deleting_an_upload_removes_both_filename_forms(temp_table):
    _add("notes.txt", 2)
    _add("upload:notes.txt", 2)

    result = _delete("notes.txt")

    assert result["chunks_deleted"] == 4
    assert store.count_rows_for("notes.txt") == 0
    assert store.count_rows_for("upload:notes.txt") == 0


def test_other_sources_are_untouched(temp_table):
    """A delete that over-reaches would destroy unrelated data."""
    _add("url:https://example.com/a", 2)
    _add("url:https://example.com/keep", 3)

    _delete("https://example.com/a")

    assert store.count_rows_for("url:https://example.com/keep") == 3


def test_deleting_a_missing_source_is_not_an_error(temp_table):
    result = _delete("https://example.com/never-stored")

    assert result["status"] == "not_found"
    assert result["chunks_deleted"] == 0
    assert "note" in result


def test_delete_is_idempotent(temp_table):
    _add("url:https://example.com/a", 2)
    first = _delete("https://example.com/a")
    second = _delete("https://example.com/a")

    assert first["chunks_deleted"] == 2
    assert second["status"] == "not_found"
    assert second["chunks_deleted"] == 0


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------


def test_dry_run_reports_without_deleting(temp_table):
    _add("url:https://example.com/a", 3)

    result = _delete("https://example.com/a", dry_run=True)

    assert result["dry_run"] is True
    assert result["chunks_to_delete"] == 3
    assert store.count_rows_for("url:https://example.com/a") == 3


def test_dry_run_shows_the_per_key_breakdown(temp_table):
    """The alias behaviour should be visible, not surprising."""
    _add("https://example.com/a", 1)
    _add("url:https://example.com/a", 2)

    result = _delete("https://example.com/a", dry_run=True)

    assert result["keys"]["url:https://example.com/a"] == 2
    assert result["keys"]["https://example.com/a"] == 1
    assert result["chunks_to_delete"] == 3


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


def test_an_incomplete_delete_is_a_failure_not_a_success(temp_table, monkeypatch):
    """The store refuses to report an unverified delete; the route must too.

    Reporting success here would repeat the `unchanged`-over-an-empty-table
    defect: a confident answer over rows that are still there.
    """
    _add("url:https://example.com/a", 2)

    def liar(source, *, replace_also=()):
        raise RuntimeError("Delete reported success but 2 rows remain")

    monkeypatch.setattr(store, "delete_source", liar)

    with pytest.raises(HTTPException) as exc:
        _delete("https://example.com/a")

    assert exc.value.status_code == 500
    assert "could not be verified" in exc.value.detail["detail"]
    assert "2 rows remain" in exc.value.detail["reason"]


def test_the_rows_are_really_gone_after_a_successful_delete(temp_table):
    """Read back through a fresh handle, the way the store verifies."""
    _add("url:https://example.com/a", 5)
    _delete("https://example.com/a")

    assert store.fresh_table().count_rows() == 0
