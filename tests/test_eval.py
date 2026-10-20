"""Tests for the multilingual evaluation harness and benchmark dataset."""

from __future__ import annotations

import pytest

from citecat.config import Config
from citecat.ingest import IngestStore
from citecat.retrieval import (
    BM25Retriever,
    DeterministicMockEmbeddingModel,
    HybridRetriever,
    SemanticRetriever,
)
from tests.eval_dataset import (
    EVAL_DATASET,
    evaluate,
    evaluate_comparison,
    populate_benchmark_corpus,
)


@pytest.fixture
def eval_store(tmp_path):
    config = Config(retrieval_db_path=str(tmp_path / "eval_multilingual.db"))
    store = IngestStore(config)
    populate_benchmark_corpus(store)
    yield store
    store.close()


def test_eval_dataset_structure():
    """Verify structure and balance of the extended multilingual evaluation benchmark."""
    assert len(EVAL_DATASET) == 14
    answerable = [q for q in EVAL_DATASET if q.answerable]
    unanswerable = [q for q in EVAL_DATASET if not q.answerable]
    assert len(answerable) == 10
    assert len(unanswerable) == 4

    # Verify coverage of French and English
    french_queries = [q for q in EVAL_DATASET if q.language == "fr"]
    english_queries = [q for q in EVAL_DATASET if q.language == "en"]
    assert len(french_queries) >= 4
    assert len(english_queries) >= 6

    # Verify specific categories exist
    categories = {q.category for q in EVAL_DATASET}
    assert "exact_match" in categories
    assert "semantic_paraphrase" in categories
    assert "cross_lingual_fr_to_en" in categories
    assert "cross_lingual_en_to_fr" in categories
    assert "accented_and_punct" in categories
    assert "multi_record" in categories
    assert "stop_words" in categories
    assert "distractor" in categories
    assert "unanswerable" in categories


def test_evaluation_bm25_baseline(eval_store):
    """Evaluate pure BM25 retrieval on the multilingual benchmark."""
    bm25 = BM25Retriever()
    metrics = evaluate(eval_store, retriever=bm25, min_score=0.4)

    assert metrics["total"] == 14
    assert metrics["unanswerable"] == 4
    # All 4 unanswerables must strictly abstain
    assert metrics["correct_abstention"] == 4
    assert metrics["false_positive_rate"] == 0.0

    # Exact matches in same language succeed
    exact_details = [d for d in metrics["details"] if d["category"] == "exact_match"]
    assert all(d["got_answer"] for d in exact_details)


def test_evaluation_hybrid_multilingual(eval_store):
    """Evaluate Hybrid retrieval (BM25 + Dense Semantic) on the benchmark."""
    model = DeterministicMockEmbeddingModel(dimension=128)
    sem_retriever = SemanticRetriever(model, store=eval_store)
    hybrid = HybridRetriever(
        lexical_retriever=BM25Retriever(),
        semantic_retriever=sem_retriever,
        alpha=0.5,
    )

    metrics = evaluate(eval_store, retriever=hybrid, min_score=0.30)

    assert metrics["total"] == 14
    # 100% correct abstention on unanswerable questions (zero hallucinations)
    assert metrics["correct_abstention"] == 4
    assert metrics["false_positive_rate"] == 0.0

    # Cross-lingual and semantic queries succeed under hybrid retrieval
    cross_lingual_fr_en = next(
        d for d in metrics["details"] if d["category"] == "cross_lingual_fr_to_en"
    )
    assert cross_lingual_fr_en["got_answer"] is True

    cross_lingual_en_fr = next(
        d for d in metrics["details"] if d["category"] == "cross_lingual_en_to_fr"
    )
    assert cross_lingual_en_fr["got_answer"] is True

    # High MRR and answerable accuracy
    assert metrics["answerable_accuracy"] >= 0.8
    assert metrics["mrr"] > 0.7


def test_evaluation_comparison_bm25_vs_hybrid(eval_store):
    """Verify side-by-side comparison demonstrating hybrid advantage on cross-lingual cases."""
    bm25 = BM25Retriever()
    model = DeterministicMockEmbeddingModel(dimension=128)
    sem_retriever = SemanticRetriever(model, store=eval_store)
    hybrid = HybridRetriever(lexical_retriever=bm25, semantic_retriever=sem_retriever, alpha=0.5)

    comp = evaluate_comparison(
        eval_store,
        lexical_retriever=bm25,
        hybrid_retriever=hybrid,
        lexical_min_score=0.4,
        hybrid_min_score=0.30,
    )

    bm25_metrics = comp["bm25"]
    hybrid_metrics = comp["hybrid"]

    # Both must preserve 100% abstention on unanswerable queries
    assert bm25_metrics["correct_abstention_rate"] == 1.0
    assert hybrid_metrics["correct_abstention_rate"] == 1.0
    assert bm25_metrics["false_positive_rate"] == 0.0
    assert hybrid_metrics["false_positive_rate"] == 0.0

    # Hybrid retrieval achieves higher or equal answerable accuracy and MRR
    assert hybrid_metrics["answerable_accuracy"] >= bm25_metrics["answerable_accuracy"]
    assert hybrid_metrics["mrr"] >= bm25_metrics["mrr"]
    assert hybrid_metrics["recall_at_k"] >= bm25_metrics["recall_at_k"]


def test_evaluation_abstract_vs_fulltext_impact(tmp_path):
    """Verify that full-text retrieval answers questions inaccessible to abstracts alone."""
    from tests.eval_dataset import (
        evaluate_abstract_vs_fulltext,
        populate_benchmark_fulltext_corpus,
    )

    abstract_store = IngestStore(Config(retrieval_db_path=str(tmp_path / "abstract.db")))
    populate_benchmark_corpus(abstract_store)

    fulltext_store = IngestStore(Config(retrieval_db_path=str(tmp_path / "fulltext.db")))
    populate_benchmark_fulltext_corpus(fulltext_store)

    bm25 = BM25Retriever()
    model = DeterministicMockEmbeddingModel(dimension=128)
    sem = SemanticRetriever(model, store=fulltext_store)
    hybrid = HybridRetriever(bm25, sem, alpha=0.5)

    try:
        results = evaluate_abstract_vs_fulltext(
            abstract_store=abstract_store,
            fulltext_store=fulltext_store,
            lexical_retriever=bm25,
            hybrid_retriever=hybrid,
            lexical_min_score=0.35,
            hybrid_min_score=0.28,
        )

        abs_bm25 = results["abstract_bm25"]
        ft_bm25 = results["fulltext_bm25"]
        ft_hybrid = results["fulltext_hybrid"]

        # Abstract-only BM25 must abstain on questions whose facts only exist in full-text
        assert abs_bm25["answerable_accuracy"] < ft_bm25["answerable_accuracy"]
        assert ft_bm25["recall_at_k"] > abs_bm25["recall_at_k"]

        # Full-text retrievers answer the full-text questions and verify citations
        assert ft_bm25["citation_verification_rate"] == 1.0
        assert ft_hybrid["citation_verification_rate"] == 1.0

        # All systems safely abstain on unanswerable control questions
        assert abs_bm25["correct_abstention_rate"] == 1.0
        assert ft_bm25["correct_abstention_rate"] == 1.0
        assert ft_hybrid["correct_abstention_rate"] == 1.0
        assert ft_bm25["false_positive_rate"] == 0.0
        assert ft_hybrid["false_positive_rate"] == 0.0

    finally:
        abstract_store.close()
        fulltext_store.close()


# ---------------------------------------------------------------------------
# Phase 5: Answer Quality & Grounded Synthesis Benchmark Tests
# ---------------------------------------------------------------------------
def test_phase5_dataset_structure():
    """Verify composition of the 18-question Phase 5 quality benchmark."""
    from tests.eval_phase5 import PHASE5_QUESTIONS

    assert len(PHASE5_QUESTIONS) == 18
    ans = [q for q in PHASE5_QUESTIONS if q.answerable]
    unans = [q for q in PHASE5_QUESTIONS if not q.answerable]
    assert len(ans) == 15
    assert len(unans) == 3

    categories = {q.category for q in PHASE5_QUESTIONS}
    assert "abstract_en" in categories
    assert "abstract_fr" in categories
    assert "pdf_fulltext" in categories
    assert "cross_lingual" in categories
    assert "unanswerable" in categories
    assert "conflicting" in categories
    assert "prompt_injection" in categories


def test_phase5_retrieval_benchmark(tmp_path):
    """Run Phase 5 evaluation on lexical BM25 retrieval."""
    from tests.eval_phase5 import evaluate_phase5_benchmark, populate_phase5_corpus

    config = Config(retrieval_db_path=str(tmp_path / "phase5_retrieval.db"), passage_min_words=5)
    store = IngestStore(config)
    populate_phase5_corpus(store)

    retriever = BM25Retriever()
    metrics = evaluate_phase5_benchmark(store, retriever, min_score=0.1, k=5)

    assert metrics["sample_size"] == 18
    assert metrics["unanswerable_count"] == 3
    # 100% citation verification
    assert metrics["citation_validity_rate"] == 1.0
    # Safe abstention on all unanswerable questions
    assert metrics["abstention_quality_rate"] == 1.0
    # Prompt injection safety
    assert metrics["injection_defense_rate"] == 1.0

    store.close()


def test_phase5_grounded_llm_synthesis_benchmark(tmp_path):
    """Run Phase 5 evaluation on GroundedSynthesizer with FakeLLMProvider."""
    from citecat.llm import FakeLLMProvider, GroundedSynthesizer
    from tests.eval_phase5 import evaluate_phase5_benchmark, populate_phase5_corpus

    config = Config(
        retrieval_db_path=str(tmp_path / "phase5_llm.db"),
        passage_min_words=5,
        synthesis_mode="llm",
    )
    store = IngestStore(config)
    populate_phase5_corpus(store)

    provider = FakeLLMProvider(mode="supported")
    synthesizer = GroundedSynthesizer(store, provider, config)

    metrics = evaluate_phase5_benchmark(store, synthesizer, min_score=0.1, k=5)

    assert metrics["sample_size"] == 18
    assert metrics["unanswerable_count"] == 3
    # All verified citations must strictly exist in source documents
    assert metrics["citation_validity_rate"] == 1.0
    # Strict abstention on unanswerable/out-of-domain questions
    assert metrics["abstention_quality_rate"] == 1.0
    # 100% defense against prompt injection payloads
    assert metrics["injection_defense_rate"] == 1.0
    # Mean latency must be measured
    assert metrics["mean_latency_seconds"] >= 0.0

    store.close()
