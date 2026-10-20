"""Citation construction and verification."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from citecat.ingest import IngestStore
from citecat.models import (
    AnswerResult,
    Citation,
    VerificationResult,
)
from citecat.retrieval import BM25Retriever, Retriever


def build_answer_with_citations(
    question: str,
    store: IngestStore,
    retriever: Retriever | None = None,
    k: int = 5,
    min_score: float = 0.5,
) -> AnswerResult:
    """Retrieve passages and construct an answer with citations.

    This baseline implementation does **not** use an LLM to generate prose
    answers.  Instead it returns the top-*k* relevant passages as
    evidence, each with a proper citation.  If no passage scores above
    *min_score*, it abstains with a clear reason.

    An LLM-based generation path can be layered on top by consuming the
    returned citations.
    """
    if retriever is None:
        retriever = BM25Retriever()

    passages = store.get_all_passages()
    if not passages:
        return AnswerResult(
            answer=None,
            citations=[],
            abstention_reason="No documents have been ingested yet.",
        )

    ranked = retriever.retrieve(question, passages, k=k)
    if not ranked:
        return AnswerResult(
            answer=None,
            citations=[],
            abstention_reason="No relevant passages found for the query.",
        )

    # Filter by minimum score
    relevant = [(p, s) for p, s in ranked if s >= min_score]
    if not relevant:
        return AnswerResult(
            answer=None,
            citations=[],
            abstention_reason=(
                f"Retrieved passages scored below the relevance threshold "
                f"({min_score}). The available evidence is insufficient to "
                f"answer the question reliably."
            ),
        )

    # Build citations from relevant passages
    citations: list[Citation] = []
    answer_parts: list[str] = []
    seen_records: set[str] = set()

    for passage, score in relevant:
        meta = store.get_record_metadata(passage.record_id)
        if meta is None:
            continue

        source_type = passage.source_type or meta.get("source_type", "abstract")
        citation = Citation(
            quote=passage.text,
            source_url=meta["source_url"],
            license=meta.get("pdf_license") or meta["license"],
            record_id=passage.record_id,
            score=round(score, 4),
            page_number=passage.page_number,
            source_type=source_type,
            pdf_url=passage.pdf_url or meta.get("pdf_url"),
        )
        citations.append(citation)

        if passage.record_id not in seen_records:
            if source_type == "pdf" and passage.page_number:
                header = f"[{meta['title']} — Page {passage.page_number}]"
            else:
                header = f"[{meta['title']}]"
            snippet = f"{passage.text[:200]}..." if len(passage.text) > 200 else passage.text
            answer_parts.append(f"{header} ({meta['source_url']}): {snippet}")
            seen_records.add(passage.record_id)

    if not citations:
        return AnswerResult(
            answer=None,
            citations=[],
            abstention_reason="Could not resolve record metadata for retrieved passages.",
        )

    answer_text = (
        f"Based on {len(citations)} source(s), here is the evidence "
        f"related to your question:\n\n" + "\n\n".join(answer_parts)
    )
    return AnswerResult(
        answer=answer_text,
        citations=citations,
        synthesis_mode="extractive",
    )


async def async_build_answer_with_citations(
    question: str,
    store: IngestStore,
    retriever: Retriever | None = None,
    k: int = 5,
    min_score: float = 0.5,
    provider: Any | None = None,
    synthesis_mode: str = "extractive",
    config: Any | None = None,
) -> AnswerResult:
    """Retrieve passages and construct an answer with citations, supporting optional LLM synthesis.

    If synthesis_mode is 'llm' and a provider is supplied, delegates to GroundedSynthesizer.
    Otherwise, executes the deterministic extractive citation builder.
    """
    if synthesis_mode.lower() == "llm" and provider is not None:
        from citecat.llm import GroundedSynthesizer

        synthesizer = GroundedSynthesizer(store=store, provider=provider, config=config)
        return await synthesizer.synthesize(question, retriever=retriever, k=k, min_score=min_score)

    return build_answer_with_citations(
        question, store, retriever=retriever, k=k, min_score=min_score
    )


def verify_citation(
    quote: str,
    source_text: str,
) -> VerificationResult:
    """Verify that *quote* appears in *source_text*.

    Verification strategy:
    1. **Exact match** – the quote appears verbatim in the source text.
    2. **Normalized match** – after Unicode normalization, whitespace
       collapsing, and case folding, check for containment.  This is a
       conservative fuzzy match that handles minor formatting differences
       without being so loose as to produce false positives.

    Returns a :class:`VerificationResult` with match details.
    """
    if not quote or not source_text:
        return VerificationResult(
            verified=False,
            detail="Empty quote or source text provided.",
        )

    # 1. Exact match
    pos = source_text.find(quote)
    if pos >= 0:
        return VerificationResult(
            verified=True,
            match_type="exact",
            match_position=pos,
            detail="Quote found verbatim in source text.",
        )

    # 2. Normalized match
    norm_quote = _normalize(quote)
    norm_source = _normalize(source_text)

    if not norm_quote:
        return VerificationResult(
            verified=False,
            detail="Quote is empty after normalization.",
        )

    pos = norm_source.find(norm_quote)
    if pos >= 0:
        return VerificationResult(
            verified=True,
            match_type="normalized",
            match_position=pos,
            detail="Quote found after normalization (whitespace/case/Unicode).",
        )

    return VerificationResult(
        verified=False,
        detail="Quote not found in source text (exact or normalized).",
    )


def _normalize(text: str) -> str:
    """Normalize text for conservative matching.

    Applies NFKC, diacritics stripping, smart quotes, lowercase, whitespace.
    """
    text = unicodedata.normalize("NFKC", text)
    # Normalize common typography variants (smart quotes, dashes, French ligatures)
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("—", "-")
        .replace("–", "-")
        .replace("œ", "oe")
        .replace("æ", "ae")
        .replace("Œ", "oe")
        .replace("Æ", "ae")
    )
    # Strip combining diacritical marks
    nfd = unicodedata.normalize("NFD", text)
    text = "".join(c for c in nfd if not unicodedata.combining(c))
    text = text.lower()
    # Normalize punctuation spacing (e.g. French space before : ; ? !)
    text = re.sub(r"\s+([:;?!])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
