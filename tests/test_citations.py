"""Tests for citecat.citations – answer construction and citation verification."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from citecat.citations import (
    _normalize,
    build_answer_with_citations,
    verify_citation,
)
from citecat.config import Config
from citecat.ingest import IngestStore
from citecat.models import Record


@pytest.fixture
def store(tmp_path):
    config = Config(
        retrieval_db_path=str(tmp_path / "test_citations.db"),
        passage_min_words=5,
        passage_max_words=50,
    )
    s = IngestStore(config)
    yield s
    s.close()


def _make_record(**overrides) -> Record:
    defaults = {
        "id": "hal-1",
        "title": "Climate Change Study",
        "snippet": "",
        "source_url": "https://hal.science/hal-1",
        "license": "cc-by-4.0",
        "retrieved_at": datetime(2024, 1, 1, tzinfo=UTC),
    }
    defaults.update(overrides)
    return Record(**defaults)


class TestBuildAnswerWithCitations:
    def test_no_ingested_documents(self, store):
        result = build_answer_with_citations("What is climate change?", store)
        assert result.answer is None
        assert result.abstention_reason is not None
        assert "ingested" in result.abstention_reason.lower()

    def test_insufficient_evidence(self, store):
        record = _make_record()
        store.ingest_record(
            record, full_text="Bananas are a tropical fruit rich in potassium and vitamins."
        )
        result = build_answer_with_citations("quantum computing algorithms", store, min_score=10.0)
        assert result.answer is None
        assert result.abstention_reason is not None

    def test_answer_with_relevant_evidence(self, store):
        record = _make_record()
        text = (
            "Climate change refers to long-term shifts in global temperatures "
            "and weather patterns. Human activities have been the main driver "
            "of climate change since the 1800s, primarily due to burning fossil "
            "fuels like coal, oil and gas. This produces heat-trapping gases. "
            "Global warming is one aspect of climate change."
        )
        store.ingest_record(record, full_text=text)
        result = build_answer_with_citations("climate change global warming", store, min_score=0.1)
        assert result.answer is not None
        assert len(result.citations) >= 1
        assert result.citations[0].source_url == "https://hal.science/hal-1"
        assert result.citations[0].license == "cc-by-4.0"
        assert result.citations[0].record_id == "hal-1"
        assert result.citations[0].score > 0
        # Quote must be exact verbatim text from the source document
        assert result.citations[0].quote in text

    def test_citations_have_real_quotes(self, store):
        record = _make_record()
        text = "Photosynthesis converts sunlight into chemical energy in plants."
        store.ingest_record(record, full_text=text)
        result = build_answer_with_citations("photosynthesis plants", store, min_score=0.1)
        assert len(result.citations) >= 1
        for c in result.citations:
            vr = verify_citation(c.quote, text)
            assert vr.verified, f"Citation quote not found in source: {c.quote[:50]}"

    def test_abstention_reason_not_empty(self, store):
        result = build_answer_with_citations("test question", store)
        assert result.abstention_reason
        assert len(result.abstention_reason) > 10


class TestVerifyCitation:
    def test_exact_match(self):
        source = "The quick brown fox jumps over the lazy dog."
        result = verify_citation("brown fox jumps", source)
        assert result.verified
        assert result.match_type == "exact"
        assert result.match_position is not None
        assert result.match_position >= 0

    def test_no_match(self):
        source = "The quick brown fox."
        result = verify_citation("blue elephant flies", source)
        assert not result.verified

    def test_normalized_match_case(self):
        source = "The Quick Brown Fox Jumps Over The Lazy Dog."
        result = verify_citation("the quick brown fox jumps", source)
        assert result.verified
        assert result.match_type == "normalized"

    def test_normalized_match_whitespace(self):
        source = "The  quick   brown\nfox jumps."
        result = verify_citation("quick brown fox", source)
        assert result.verified
        assert result.match_type == "normalized"

    def test_empty_quote(self):
        result = verify_citation("", "some source text")
        assert not result.verified

    def test_empty_source(self):
        result = verify_citation("some quote", "")
        assert not result.verified

    def test_fabricated_quote_rejected(self):
        source = (
            "Machine learning models require large datasets for training. "
            "Deep learning is a subset of machine learning."
        )
        result = verify_citation(
            "Machine learning is the best technology ever invented",
            source,
        )
        assert not result.verified

    def test_partial_fabrication_rejected(self):
        source = "Climate change is caused by greenhouse gas emissions."
        result = verify_citation(
            "Climate change is caused by solar radiation changes.",
            source,
        )
        assert not result.verified

    def test_unicode_normalization(self):
        source = "café résumé naïve"
        result = verify_citation("cafe resume naive", source)
        assert result.verified is True
        assert result.match_type == "normalized"

    def test_typography_normalization(self):
        source = "The study stated: “Open-access science increases citation counts.”"
        quote = 'The study stated: "Open-access science increases citation counts."'
        result = verify_citation(quote, source)
        assert result.verified is True
        assert result.match_type == "normalized"

    def test_ambiguous_repeated_quote_matches_first_position(self):
        """Repeated phrase matches and returns the initial occurrence position."""
        source = (
            "Ocean acidification affects marine ecosystems. "
            "Recent studies confirm Ocean acidification affects marine ecosystems worldwide."
        )
        quote = "Ocean acidification affects marine ecosystems"
        result = verify_citation(quote, source)
        assert result.verified is True
        assert result.match_position == 0

    def test_partial_sentence_with_punctuation_boundaries(self):
        """Partial phrase within complex punctuation matches reliably."""
        source = (
            "The experimental results (see Figure 3, p. 45; Table 1) "
            "indicated a 3.4% enhancement: a statistically significant margin."
        )
        quote = "indicated a 3.4% enhancement: a statistically significant margin."
        result = verify_citation(quote, source)
        assert result.verified is True
        assert result.match_type == "exact"

    def test_french_ligatures_and_cedilla_normalization(self):
        """French œ, æ, and ç normalize to match ascii or decomposed variants."""
        source = "Une étude sur le cœur des écosystèmes et la façon d'agir pour le curriculum vitæ."
        quote = (
            "Une etude sur le coeur des ecosystemes et la facon d'agir pour le curriculum vitae."
        )
        result = verify_citation(quote, source)
        assert result.verified is True
        assert result.match_type == "normalized"

    def test_french_narrow_non_breaking_space(self):
        """Narrow non-breaking spaces before punctuation normalize seamlessly."""
        # \u202f is French narrow no-break space before colon and question mark
        source = "Résultats remarquables\u202f: 42% d'amélioration\u202f?"
        quote = "resultats remarquables: 42% d'amelioration?"
        result = verify_citation(quote, source)
        assert result.verified is True
        assert result.match_type == "normalized"


class TestNormalize:
    def test_lowercase(self):
        assert _normalize("HELLO") == "hello"

    def test_whitespace_collapse(self):
        assert _normalize("  hello   world  ") == "hello world"

    def test_newlines(self):
        assert _normalize("hello\n\nworld") == "hello world"

    def test_ligature_normalization(self):
        assert _normalize("œuvre et curriculum vitæ") == "oeuvre et curriculum vitae"


class TestClaimToCitationProvenanceAudit:
    """Audit end-to-end mapping from synthesized claims to verified citations."""

    @pytest.mark.asyncio
    async def test_all_displayed_claims_map_to_verified_citations(self, tmp_path):
        from citecat.citations import async_build_answer_with_citations
        from citecat.llm import FakeLLMProvider
        from citecat.models import ExtractedPage, PDFExtractionResult

        config = Config(
            retrieval_db_path=str(tmp_path / "claim_audit.db"),
            passage_min_words=5,
            synthesis_mode="llm",
        )
        store = IngestStore(config)

        # Ingest a PDF document with page provenance
        rec = Record(
            id="hal-audit-1",
            title="Marine Heatwaves in the Mediterranean",
            snippet="Abstract on marine heatwaves.",
            source_url="https://hal.science/hal-audit-1",
            license="https://creativecommons.org/licenses/by/4.0/",
            retrieved_at=datetime.now(UTC),
            pdf_url="https://hal.science/hal-audit-1/document",
        )
        extraction = PDFExtractionResult(
            record_id="hal-audit-1",
            pdf_url=rec.pdf_url,
            status="success",
            total_pages=1,
            extracted_pages=1,
            total_words=20,
            ocr_required=False,
            pages=[
                ExtractedPage(
                    page_number=1,
                    text="Marine heatwaves in summer 2022 broke all previous thermal records.",
                    word_count=10,
                )
            ],
            full_text="Marine heatwaves in summer 2022 broke all previous thermal records.",
        )
        store.ingest_pdf(rec, extraction)

        provider = FakeLLMProvider(mode="supported")
        result = await async_build_answer_with_citations(
            "Marine heatwaves 2022 thermal records",
            store,
            synthesis_mode="llm",
            provider=provider,
            config=config,
            min_score=0.1,
        )

        assert result.answer is not None
        assert len(result.claims) >= 1
        assert len(result.citations) >= 1

        # Confirm every claim's passage_ids map to returned citations
        citation_passage_record_ids = {c.record_id for c in result.citations}
        for claim in result.claims:
            assert len(claim.passage_ids) >= 1
            assert all(c.record_id in citation_passage_record_ids for c in result.citations)
            # Stored text verification
            stored_text = store.get_record_text("hal-audit-1")
            if claim.quote:
                vr = verify_citation(claim.quote, stored_text)
                assert vr.verified is True

        # Confirm metadata provenance comes strictly from store
        cit = result.citations[0]
        assert cit.record_id == "hal-audit-1"
        assert cit.source_url == "https://hal.science/hal-audit-1"
        assert cit.page_number == 1
        assert cit.source_type == "pdf"
        assert cit.pdf_url == "https://hal.science/hal-audit-1/document"
        assert cit.license == "https://creativecommons.org/licenses/by/4.0/"

        store.close()
