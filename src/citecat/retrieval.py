"""Lexical retrieval baseline with a pluggable interface for future embedding-based search."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from citecat.models import Passage


class Retriever(Protocol):
    """Interface for passage retrieval.

    Implementing classes must accept a list of passages and a query,
    returning scored results.  This allows swapping the lexical baseline
    for vector search later.
    """

    def retrieve(
        self, query: str, passages: list[Passage], k: int = 5
    ) -> list[tuple[Passage, float]]:
        """Return the top-*k* passages ranked by relevance score."""
        ...


_STOPWORDS = {
    # English stop words
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
    # French stop words
    "de",
    "la",
    "le",
    "les",
    "un",
    "une",
    "des",
    "du",
    "et",
    "en",
    "est",
    "sont",
    "dans",
    "pour",
    "par",
    "qui",
    "que",
    "sur",
    "avec",
    "ce",
    "cette",
    "ces",
    "au",
    "aux",
    "ne",
    "pas",
    "plus",
    "son",
    "sa",
    "ses",
    "leur",
    "leurs",
    "ou",
    "ont",
    "d",
    "l",
    "c",
    "s",
    "n",
    "j",
    "m",
    "t",
    "qu",
    "quel",
    "quelle",
    "quels",
    "quelles",
}


class BM25Retriever:
    """BM25-based lexical retriever with stop-word filtering.

    This is a self-contained implementation that scores each passage
    against the query using the Okapi BM25 formula.  It requires no
    external dependencies or paid APIs.

    Parameters:
        k1: Term-frequency saturation parameter (default 1.5).
        b: Length-normalization parameter (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def retrieve(
        self, query: str, passages: list[Passage], k: int = 5
    ) -> list[tuple[Passage, float]]:
        """Score all passages against *query* with BM25 and return top-*k*."""
        if not passages or not query.strip():
            return []

        all_query_terms = _tokenize(query)
        # Filter stopwords, but fall back to raw tokens if all were stopwords
        query_terms = [t for t in all_query_terms if t not in _STOPWORDS]
        if not query_terms:
            query_terms = all_query_terms
        if not query_terms:
            return []

        # Pre-compute corpus statistics
        doc_freqs: Counter[str] = Counter()
        doc_tokens: list[list[str]] = []
        total_length = 0

        for p in passages:
            raw_tokens = _tokenize(p.text)
            tokens = [t for t in raw_tokens if t not in _STOPWORDS]
            doc_tokens.append(tokens)
            total_length += len(tokens)
            for term in set(tokens):
                doc_freqs[term] += 1

        n = len(passages)
        avg_dl = total_length / n if n else 1.0

        scored: list[tuple[Passage, float]] = []
        for i, p in enumerate(passages):
            tokens = doc_tokens[i]
            tf_map = Counter(tokens)
            dl = len(tokens)
            score = 0.0

            for qt in query_terms:
                if qt not in doc_freqs:
                    continue
                df = doc_freqs[qt]
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                tf = tf_map.get(qt, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / avg_dl)
                score += idf * numerator / denominator

            scored.append((p, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"\b\w+\b", text.lower())


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two float vectors.

    Uses numpy if installed, falling back to pure Python math.
    Returns a float in [-1.0, 1.0].
    """
    if len(a) != len(b) or not a:
        return 0.0

    try:
        import numpy as np

        arr_a = np.asarray(a, dtype=np.float32)
        arr_b = np.asarray(b, dtype=np.float32)
        norm_a = float(np.linalg.norm(arr_a))
        norm_b = float(np.linalg.norm(arr_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(arr_a, arr_b) / (norm_a * norm_b))
    except ImportError:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


class EmbeddingModel(Protocol):
    """Protocol for embedding models."""

    @property
    def model_name(self) -> str:
        """Identifier of the model."""
        ...

    @property
    def dimension(self) -> int:
        """Output embedding vector dimension."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query into a dense vector."""
        ...

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of passage texts into dense vectors."""
        ...


class SentenceTransformerEmbeddingModel:
    """Multilingual dense embedding model backed by sentence-transformers."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        cache_folder: str | None = None,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._cache_folder = cache_folder
        self._device = device
        self._model = None
        self._dimension: int | None = None

    def _ensure_model(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as err:
                raise ImportError(
                    "sentence-transformers is required for neural semantic retrieval. "
                    "Install with: uv sync --extra semantic or pip install 'citecat[semantic]'"
                ) from err

            import logging

            logger = logging.getLogger(__name__)
            logger.info("Loading sentence transformer model '%s'...", self._model_name)
            self._model = SentenceTransformer(
                self._model_name,
                cache_folder=self._cache_folder,
                device=self._device,
            )
            sample = self._model.encode(["test"], show_progress_bar=False)
            self._dimension = len(sample[0])

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        self._ensure_model()
        return self._dimension or 1024

    def embed_query(self, text: str) -> list[float]:
        self._ensure_model()
        vec = self._model.encode(text, show_progress_bar=False, normalize_embeddings=True)
        return [float(x) for x in vec]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_model()
        vecs = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return [[float(x) for x in v] for v in vecs]


class DeterministicMockEmbeddingModel:
    """Fast, deterministic multilingual embedding model for testing and offline evaluation.

    Encodes French and English semantic concepts into a shared vector space
    using bilingual synsets and n-gram hash projections.
    Requires no network access, GPUs, or external dependencies.
    """

    def __init__(self, dimension: int = 128, model_name: str = "mock-multilingual-v1") -> None:
        self._dimension = dimension
        self._model_name = model_name
        # Bilingual concept synsets for cross-lingual semantic matching
        self._synsets = [
            # Climate & biodiversity
            {
                "climat",
                "climate",
                "réchauffement",
                "warming",
                "biodiversité",
                "biodiversity",
                "écosystème",
                "ecosystem",
                "terrestre",
                "terrestrial",
                "marin",
                "marine",
            },
            # AI, ML, NLP
            {
                "apprentissage",
                "machine",
                "learning",
                "langage",
                "language",
                "nlp",
                "automatique",
                "transformer",
                "modèle",
                "model",
                "traduction",
                "intelligence",
                "artificielle",
            },
            # Biology, protein
            {
                "protéine",
                "protein",
                "acide",
                "aminé",
                "amino",
                "acid",
                "structure",
                "prédire",
                "prediction",
                "biologie",
                "biology",
                "tridimensionnelle",
            },
            # Quantum physics
            {
                "quantique",
                "quantum",
                "superposition",
                "intrication",
                "entanglement",
                "ordinateur",
                "computer",
                "cryptographie",
                "cryptography",
            },
            # Medicine & health
            {
                "santé",
                "health",
                "médical",
                "medical",
                "maladie",
                "disease",
                "thérapie",
                "therapy",
                "traitement",
                "treatment",
            },
        ]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, text: str) -> list[float]:
        import hashlib

        tokens = set(re.findall(r"\b\w+\b", text.lower()))
        vec = [0.0] * self._dimension

        # 1. Project bilingual semantic synsets into coordinate clusters
        chunk_size = max(1, self._dimension // (len(self._synsets) + 1))
        for s_idx, synset in enumerate(self._synsets):
            overlap = len(tokens.intersection(synset))
            if overlap > 0:
                base = s_idx * chunk_size
                for offset in range(chunk_size):
                    idx = (base + offset) % self._dimension
                    vec[idx] += 3.0 * overlap

        # 2. Hash-based n-gram projection for general lexical/semantic similarity
        words = list(tokens)
        for word in words:
            for i in range(len(word) - 2):
                ngram = word[i : i + 3]
                h = int(hashlib.md5(ngram.encode("utf-8")).hexdigest()[:8], 16)
                idx = h % self._dimension
                sign = 1.0 if (h & 1) else -1.0
                vec[idx] += sign * 0.5

        # Normalize to unit length (L2 norm)
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        else:
            vec[0] = 1.0
        return vec

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


class SemanticRetriever:
    """Passage retriever using dense embedding cosine similarity."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        store: object | None = None,
    ) -> None:
        self.model = embedding_model
        self.store = store

    def retrieve(
        self, query: str, passages: list[Passage], k: int = 5
    ) -> list[tuple[Passage, float]]:
        if not passages or not query.strip():
            return []

        query_vec = self.model.embed_query(query)

        # Retrieve cached embeddings from store if available
        passage_vecs: dict[tuple[str, int, int], list[float]] = {}
        if self.store is not None and hasattr(self.store, "get_embeddings_for_passages"):
            passage_vecs = self.store.get_embeddings_for_passages(
                passages, self.model.model_name, expected_dim=self.model.dimension
            )

        # Identify passages needing embedding
        missing_passages = [
            p for p in passages if (p.record_id, p.start_char, p.end_char) not in passage_vecs
        ]

        if missing_passages:
            missing_texts = [p.text for p in missing_passages]
            new_vecs = self.model.embed_passages(missing_texts)

            to_store: list[tuple[str, int, int, str, list[float]]] = []
            for p, vec in zip(missing_passages, new_vecs, strict=True):
                key = (p.record_id, p.start_char, p.end_char)
                passage_vecs[key] = vec
                to_store.append((p.record_id, p.start_char, p.end_char, p.text, vec))

            if (
                self.store is not None
                and hasattr(self.store, "store_embeddings_batch")
                and to_store
            ):
                self.store.store_embeddings_batch(to_store, self.model.model_name)

        scored: list[tuple[Passage, float]] = []
        for p in passages:
            key = (p.record_id, p.start_char, p.end_char)
            p_vec = passage_vecs.get(key)
            if p_vec is None:
                sim = 0.0
            else:
                raw_sim = cosine_similarity(query_vec, p_vec)
                # Clamp cosine similarity to [0.0, 1.0] for non-negative scoring
                sim = max(0.0, raw_sim)
            scored.append((p, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


class HybridRetriever:
    """Hybrid retriever combining lexical BM25 and semantic dense retrieval.

    Calculates calibrated normalized scores:
        score_hybrid = alpha * s_lex_norm + (1 - alpha) * s_sem_norm
    where s_lex_norm is BM25 calibrated into [0, 1] and s_sem_norm is
    clamped cosine similarity in [0, 1].

    Parameters:
        lexical_retriever: BM25Retriever instance.
        semantic_retriever: SemanticRetriever instance.
        alpha: Weight for lexical score (default 0.5; 1.0 = BM25 only, 0.0 = semantic only).
    """

    def __init__(
        self,
        lexical_retriever: BM25Retriever | None = None,
        semantic_retriever: SemanticRetriever | None = None,
        alpha: float = 0.5,
    ) -> None:
        self.lexical_retriever = lexical_retriever or BM25Retriever()
        self.semantic_retriever = semantic_retriever
        self.alpha = max(0.0, min(1.0, alpha))

    def retrieve(
        self, query: str, passages: list[Passage], k: int = 5
    ) -> list[tuple[Passage, float]]:
        if not passages or not query.strip():
            return []

        # If alpha is 1.0 or semantic retriever is missing, purely lexical
        if self.alpha == 1.0 or self.semantic_retriever is None:
            return self.lexical_retriever.retrieve(query, passages, k)

        import logging

        logger = logging.getLogger(__name__)

        # If alpha is 0.0, purely semantic
        if self.alpha == 0.0:
            try:
                return self.semantic_retriever.retrieve(query, passages, k)
            except Exception as exc:
                logger.warning("Semantic retrieval failed (%s), falling back to lexical BM25.", exc)
                return self.lexical_retriever.retrieve(query, passages, k)

        # Hybrid: retrieve lexical and semantic scores
        # 1. Lexical BM25 scores for all passages
        lex_results = self.lexical_retriever.retrieve(query, passages, k=len(passages))
        lex_map = {(p.record_id, p.start_char, p.end_char): score for p, score in lex_results}

        # 2. Semantic scores for all passages with graceful fallback
        sem_map: dict[tuple[str, int, int], float] = {}
        try:
            sem_results = self.semantic_retriever.retrieve(query, passages, k=len(passages))
            sem_map = {(p.record_id, p.start_char, p.end_char): score for p, score in sem_results}
        except Exception as exc:
            logger.warning(
                "Semantic retrieval failed during hybrid search (%s), falling back to BM25.", exc
            )
            return self.lexical_retriever.retrieve(query, passages, k)

        # 3. Score Normalization & Calibration
        max_lex = max(lex_map.values()) if lex_map else 0.0

        hybrid_scored: list[tuple[Passage, float]] = []
        for p in passages:
            key = (p.record_id, p.start_char, p.end_char)
            raw_lex = lex_map.get(key, 0.0)
            raw_sem = sem_map.get(key, 0.0)

            # Normalize BM25 to [0, 1] relative to corpus maximum for this query
            norm_lex = (raw_lex / max_lex) if max_lex > 0 else 0.0
            # Semantic cosine similarity is already clamped to [0, 1]
            norm_sem = max(0.0, min(1.0, raw_sem))

            hybrid_score = self.alpha * norm_lex + (1.0 - self.alpha) * norm_sem
            hybrid_scored.append((p, hybrid_score))

        hybrid_scored.sort(key=lambda x: x[1], reverse=True)
        return hybrid_scored[:k]
