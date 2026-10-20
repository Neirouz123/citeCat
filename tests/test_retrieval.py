"""Tests for citecat.retrieval – BM25 retriever."""

from __future__ import annotations

from citecat.models import Passage
from citecat.retrieval import BM25Retriever, _tokenize


def _passage(text: str, record_id: str = "hal-1") -> Passage:
    return Passage(
        record_id=record_id,
        text=text,
        start_char=0,
        end_char=len(text),
        word_count=len(text.split()),
    )


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("Hello World!")
        assert tokens == ["hello", "world"]

    def test_empty(self):
        assert _tokenize("") == []

    def test_numbers(self):
        tokens = _tokenize("Python 3.11 is great")
        assert "python" in tokens
        assert "3" in tokens
        assert "11" in tokens


class TestBM25Retriever:
    def test_empty_passages(self):
        r = BM25Retriever()
        result = r.retrieve("test", [], k=5)
        assert result == []

    def test_empty_query(self):
        r = BM25Retriever()
        passages = [_passage("some text here")]
        result = r.retrieve("", passages, k=5)
        assert result == []

    def test_single_passage_match(self):
        r = BM25Retriever()
        passages = [_passage("machine learning is a subfield of artificial intelligence")]
        result = r.retrieve("machine learning", passages, k=5)
        assert len(result) == 1
        assert result[0][1] > 0  # positive score

    def test_ranking_order(self):
        r = BM25Retriever()
        passages = [
            _passage("climate change and global warming effects on biodiversity", "hal-1"),
            _passage("machine learning algorithms for natural language processing", "hal-2"),
            _passage("climate models predict global warming acceleration in tropics", "hal-3"),
        ]
        result = r.retrieve("climate change global warming", passages, k=3)
        # The climate-related passages should rank higher
        assert result[0][0].record_id in ("hal-1", "hal-3")
        assert result[1][0].record_id in ("hal-1", "hal-3")

    def test_k_limits_results(self):
        r = BM25Retriever()
        passages = [_passage(f"document number {i} about science") for i in range(20)]
        result = r.retrieve("science", passages, k=3)
        assert len(result) == 3

    def test_no_matching_terms(self):
        r = BM25Retriever()
        passages = [_passage("alpha bravo charlie")]
        result = r.retrieve("xyz quantum", passages, k=5)
        # Score should be 0 or very low since no terms match
        assert len(result) == 1
        assert result[0][1] == 0.0


class TestCosineSimilarity:
    def test_identical_vectors(self):
        from citecat.retrieval import cosine_similarity

        vec = [1.0, 2.0, 3.0]
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        from citecat.retrieval import cosine_similarity

        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]
        sim = cosine_similarity(vec_a, vec_b)
        assert abs(sim) < 1e-5

    def test_zero_vector(self):
        from citecat.retrieval import cosine_similarity

        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_dimension_mismatch(self):
        from citecat.retrieval import cosine_similarity

        assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


class TestDeterministicMockEmbeddingModel:
    def test_dimension_and_norm(self):
        import math

        from citecat.retrieval import DeterministicMockEmbeddingModel

        model = DeterministicMockEmbeddingModel(dimension=64)
        assert model.dimension == 64
        assert model.model_name == "mock-multilingual-v1"

        vec = model.embed_query("quantum computing algorithms")
        assert len(vec) == 64
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-4

    def test_cross_lingual_similarity(self):
        from citecat.retrieval import DeterministicMockEmbeddingModel, cosine_similarity

        model = DeterministicMockEmbeddingModel(dimension=128)
        vec_fr = model.embed_query("changement climatique et biodiversité")
        vec_en = model.embed_query("climate change impacts on biodiversity")
        vec_unrelated = model.embed_query("recette du gâteau au chocolat")

        sim_related = cosine_similarity(vec_fr, vec_en)
        sim_unrelated = cosine_similarity(vec_fr, vec_unrelated)

        assert sim_related > 0.5
        assert sim_related > sim_unrelated


class TestSemanticRetriever:
    def test_semantic_ranking(self):
        from citecat.retrieval import DeterministicMockEmbeddingModel, SemanticRetriever

        model = DeterministicMockEmbeddingModel(dimension=128)
        retriever = SemanticRetriever(model)

        passages = [
            _passage("Global climate change leads to biodiversity loss", "hal-climate"),
            _passage("Transformers revolutionized natural language processing", "hal-nlp"),
        ]

        results = retriever.retrieve("climat et réchauffement", passages, k=2)
        assert len(results) == 2
        assert results[0][0].record_id == "hal-climate"
        assert results[0][1] > results[1][1]

    def test_empty_query_and_passages(self):
        from citecat.retrieval import DeterministicMockEmbeddingModel, SemanticRetriever

        model = DeterministicMockEmbeddingModel(dimension=64)
        retriever = SemanticRetriever(model)

        assert retriever.retrieve("", [_passage("test")]) == []
        assert retriever.retrieve("test", []) == []


class TestHybridRetriever:
    def test_pure_lexical_when_alpha_one(self):
        from citecat.retrieval import BM25Retriever, HybridRetriever

        hybrid = HybridRetriever(lexical_retriever=BM25Retriever(), alpha=1.0)
        passages = [_passage("deep learning natural language processing")]
        res = hybrid.retrieve("deep learning", passages)
        assert len(res) == 1
        assert res[0][1] > 0

    def test_calibrated_scoring_range(self):
        from citecat.retrieval import (
            BM25Retriever,
            DeterministicMockEmbeddingModel,
            HybridRetriever,
            SemanticRetriever,
        )

        model = DeterministicMockEmbeddingModel(dimension=128)
        sem = SemanticRetriever(model)
        hybrid = HybridRetriever(
            lexical_retriever=BM25Retriever(),
            semantic_retriever=sem,
            alpha=0.5,
        )

        passages = [
            _passage("Climate change and terrestrial biodiversity extinction risk", "hal-1"),
            _passage("Neural network language models for syntactic parsing", "hal-2"),
        ]

        results = hybrid.retrieve("climate warming", passages, k=2)
        assert len(results) == 2
        # All hybrid scores must be calibrated in [0.0, 1.0]
        for _, score in results:
            assert 0.0 <= score <= 1.0

    def test_graceful_fallback_on_semantic_failure(self):
        from citecat.retrieval import BM25Retriever, HybridRetriever

        class FailingSemanticRetriever:
            def retrieve(self, query, passages, k=5):
                raise RuntimeError("GPU out of memory or model unavailable")

        hybrid = HybridRetriever(
            lexical_retriever=BM25Retriever(),
            semantic_retriever=FailingSemanticRetriever(),  # type: ignore
            alpha=0.5,
        )

        passages = [_passage("quantum computing entanglement")]
        # Should catch error, log warning, and fall back to BM25
        results = hybrid.retrieve("quantum", passages, k=1)
        assert len(results) == 1
        assert results[0][1] > 0
