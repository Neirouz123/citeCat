"""Tests for citecat.ingest – passage splitting and ingestion store."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from citecat.config import Config
from citecat.ingest import IngestStore, split_into_passages
from citecat.models import Record


@pytest.fixture
def store(tmp_path):
    config = Config(
        retrieval_db_path=str(tmp_path / "test_retrieval.db"),
        passage_min_words=5,
        passage_max_words=10,
    )
    s = IngestStore(config)
    yield s
    s.close()


def _make_record(**overrides) -> Record:
    defaults = {
        "id": "hal-1",
        "title": "Test",
        "snippet": "",
        "source_url": "https://hal.science/hal-1",
        "license": "cc-by",
        "retrieved_at": datetime(2024, 1, 1, tzinfo=UTC),
    }
    defaults.update(overrides)
    return Record(**defaults)


class TestSplitIntoPassages:
    def test_empty_text(self):
        assert split_into_passages("") == []

    def test_whitespace_only(self):
        assert split_into_passages("   ") == []

    def test_short_text_single_passage(self):
        text = "word " * 8
        passages = split_into_passages(text.strip(), min_words=5, max_words=10)
        assert len(passages) == 1
        _, start, end, wc = passages[0]
        assert wc == 8
        assert start == 0

    def test_long_text_multiple_passages(self):
        text = " ".join(f"word{i}" for i in range(25))
        passages = split_into_passages(text, min_words=5, max_words=10)
        assert len(passages) >= 2
        total_words = sum(wc for _, _, _, wc in passages)
        assert total_words == 25

    def test_passage_offsets_cover_text(self):
        words = [f"w{i}" for i in range(15)]
        text = " ".join(words)
        passages = split_into_passages(text, min_words=5, max_words=10)
        # Each passage text must strictly match the slice of source text
        for p_text, start, end, _wc in passages:
            assert text[start:end] == p_text

    def test_passage_offsets_with_repeated_words(self):
        text = "the cat and the dog and the bird and the"
        passages = split_into_passages(text, min_words=2, max_words=5)
        assert len(passages) >= 2
        for p_text, start, end, _wc in passages:
            assert text[start:end] == p_text

    def test_sentence_boundary_splitting(self):
        # Build text where a sentence ends mid-range
        words = ["hello"] * 6 + ["world."] + ["foo"] * 5
        text = " ".join(words)
        passages = split_into_passages(text, min_words=5, max_words=10)
        assert passages[0][0].endswith("world.")


class TestIngestStore:
    def test_ingest_and_retrieve(self, store):
        snippet = "Alpha bravo charlie delta echo foxtrot golf hotel india juliet."
        record = _make_record(snippet=snippet)
        count = store.ingest_record(record)
        assert count >= 1
        passages = store.get_all_passages()
        assert len(passages) >= 1
        assert passages[0].record_id == "hal-1"

    def test_ingest_replaces_old_passages(self, store):
        record = _make_record(snippet="One two three four five six seven eight nine ten.")
        store.ingest_record(record)
        old = store.get_all_passages()
        # Re-ingest with different text
        full_text = "New text with different words completely updated fresh."
        store.ingest_record(record, full_text=full_text)
        new = store.get_all_passages()
        assert len(new) >= 1
        assert new[0].text != old[0].text

    def test_get_record_metadata(self, store):
        record = _make_record()
        store.ingest_record(record, full_text="Some text here for testing.")
        meta = store.get_record_metadata("hal-1")
        assert meta is not None
        assert meta["source_url"] == "https://hal.science/hal-1"
        assert meta["license"] == "cc-by"

    def test_get_record_text(self, store):
        record = _make_record()
        store.ingest_record(record, full_text="Full text content.")
        text = store.get_record_text("hal-1")
        assert text == "Full text content."

    def test_missing_record_metadata(self, store):
        assert store.get_record_metadata("nonexistent") is None


class TestEmbeddingStorage:
    def test_store_and_retrieve_embedding(self, store):
        record = _make_record(snippet="Quantum mechanics principles.")
        store.ingest_record(record)
        passages = store.get_all_passages()
        p = passages[0]

        vec = [0.123, 0.456, 0.789]
        store.store_embedding(p.record_id, p.start_char, p.end_char, p.text, "test-model", vec)

        retrieved = store.get_embedding(
            p.record_id, p.start_char, p.end_char, p.text, "test-model", expected_dim=3
        )
        assert retrieved is not None
        assert len(retrieved) == 3
        for val, ret_val in zip(vec, retrieved, strict=True):
            assert abs(val - ret_val) < 1e-5

    def test_stale_embedding_detection_on_text_change(self, store):
        record = _make_record(snippet="Original text for embedding.")
        store.ingest_record(record)
        passages = store.get_all_passages()
        p = passages[0]

        vec = [0.5, 0.5]
        store.store_embedding(p.record_id, p.start_char, p.end_char, p.text, "test-model", vec)

        # Query with altered text
        stale_check = store.get_embedding(
            p.record_id, p.start_char, p.end_char, "Altered text completely different", "test-model"
        )
        assert stale_check is None

    def test_corrupt_blob_handled_gracefully(self, store):
        conn = store._get_conn()
        conn.execute(
            """
            INSERT INTO embeddings
            (record_id, start_char, end_char, text_hash, model_name, dimension, vector, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("hal-1", 0, 10, "fakehash", "test-model", 4, b"not_4_floats", "2024-01-01"),
        )
        conn.commit()

        # Should return None and not crash
        res = store.get_embedding("hal-1", 0, 10, "fake_text", "test-model")
        assert res is None

    def test_batch_storage_and_passage_lookup(self, store):
        record = _make_record(snippet="First passage. Second passage.")
        store.ingest_record(record)
        passages = store.get_all_passages()
        assert len(passages) >= 1

        batch = [
            (p.record_id, p.start_char, p.end_char, p.text, [0.1 * i, 0.2 * i])
            for i, p in enumerate(passages, start=1)
        ]
        store.store_embeddings_batch(batch, "batch-model")

        cached = store.get_embeddings_for_passages(passages, "batch-model", expected_dim=2)
        assert len(cached) == len(passages)
        assert store.count_embeddings("batch-model") == len(passages)

    def test_delete_stale_embeddings(self, store):
        record = _make_record(snippet="Old passage text.")
        store.ingest_record(record)
        passages = store.get_all_passages()
        p = passages[0]

        store.store_embedding(
            p.record_id, p.start_char, p.end_char, p.text, "test-model", [1.0, 2.0]
        )
        assert store.count_embeddings("test-model") == 1

        # Re-ingest record with different text, which changes passage boundaries and text
        record_new = _make_record(snippet="Completely renewed different content for passage.")
        store.ingest_record(record_new)

        # Old embeddings for this record are automatically removed on re-ingest
        assert store.count_embeddings("test-model") == 0
