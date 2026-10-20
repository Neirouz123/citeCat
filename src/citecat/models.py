"""Pydantic models for CiteCat records and citations."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class Record(BaseModel):
    """A validated HAL publication record.

    Every record **must** have an ``id``, ``title``, ``snippet``,
    ``source_url``, ``license``, and ``retrieved_at``.  Optional fields
    include ``authors``, ``language``, and ``publication_date``.

    ``source_url`` and ``license`` are required to ensure citation
    provenance.  Unknown licenses are represented as ``"unknown"``
    rather than silently assumed.
    """

    id: str = Field(..., description="HAL document identifier (e.g. 'hal-01234567')")
    title: str = Field(..., min_length=1, description="Publication title")
    snippet: str = Field(default="", description="Short text excerpt or abstract")
    source_url: str = Field(
        ..., min_length=1, description="Canonical URL on HAL (must not be invented)"
    )
    license: str = Field(..., min_length=1, description="License identifier or 'unknown'")
    retrieved_at: datetime = Field(..., description="UTC timestamp of retrieval")

    # Optional metadata
    authors: list[str] = Field(default_factory=list, description="Author full names")
    language: str | None = Field(default=None, description="ISO 639 language code")
    publication_date: str | None = Field(
        default=None, description="Publication date as returned by HAL"
    )
    pdf_url: str | None = Field(default=None, description="Open-access PDF URL if available")
    pdf_license: str | None = Field(
        default=None, description="License for deposited PDF if specified"
    )
    files: list[str] = Field(default_factory=list, description="List of deposited file URLs on HAL")
    open_access: bool = Field(default=False, description="Whether document is flagged open access")

    @field_validator("source_url")
    @classmethod
    def source_url_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("source_url must not be empty or whitespace")
        return v

    @field_validator("license")
    @classmethod
    def license_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("license must not be empty or whitespace")
        return v


class Passage(BaseModel):
    """A text passage extracted from a record for retrieval."""

    record_id: str
    text: str
    start_char: int = Field(..., ge=0, description="Character offset in source text")
    end_char: int = Field(..., ge=0)
    word_count: int = Field(..., ge=0)
    page_number: int | None = Field(
        default=None, ge=1, description="1-indexed page number if extracted from PDF"
    )
    source_type: str = Field(default="abstract", description="'abstract' or 'pdf'")
    pdf_url: str | None = Field(default=None, description="Original PDF URL if from PDF")


class Citation(BaseModel):
    """A citation linking a claim to its source evidence."""

    quote: str = Field(..., description="Exact quoted text from the source")
    source_url: str = Field(..., description="Canonical source URL")
    license: str = Field(..., description="License of the source material")
    record_id: str = Field(..., description="HAL record identifier")
    score: float = Field(..., description="Retrieval score (higher = more relevant)")
    page_number: int | None = Field(
        default=None, ge=1, description="1-indexed page number if from PDF"
    )
    source_type: str = Field(default="abstract", description="'abstract' or 'pdf'")
    pdf_url: str | None = Field(default=None, description="Original PDF URL if from PDF")


class EvidencePassage(BaseModel):
    """An evidence passage presented to the LLM for grounded answer generation."""

    id: str = Field(..., description="Stable short identifier, e.g. 'P1', 'P2'")
    record_id: str = Field(..., description="HAL document identifier")
    title: str = Field(..., description="Publication title")
    text: str = Field(..., description="Verbatim passage text")
    page_number: int | None = Field(
        default=None, ge=1, description="1-indexed page number if from PDF"
    )
    source_url: str = Field(..., description="Canonical source URL")
    license: str = Field(..., description="Document license")
    score: float = Field(..., description="Retrieval relevance score")
    source_type: str = Field(default="abstract", description="'abstract' or 'pdf'")
    pdf_url: str | None = Field(default=None, description="Original PDF URL if from PDF")


class GroundedClaim(BaseModel):
    """A factual claim synthesized by an LLM with cited passage identifiers."""

    claim: str = Field(..., description="The synthesized factual claim")
    passage_ids: list[str] = Field(
        default_factory=list, description="IDs of supporting passages, e.g. ['P1']"
    )
    quote: str | None = Field(
        default=None, description="Verbatim quote from the supporting passage"
    )


class LLMSynthesisOutput(BaseModel):
    """Raw structured output produced by an LLM provider."""

    answer: str = Field(..., description="Synthesized narrative answer")
    claims: list[GroundedClaim] = Field(
        default_factory=list, description="List of individual grounded claims"
    )
    abstain: bool = Field(default=False, description="True if evidence is insufficient to answer")
    abstention_reason: str | None = Field(
        default=None, description="Reason for abstaining if evidence is insufficient"
    )


class AnswerResult(BaseModel):
    """Structured result from answer_with_citations."""

    answer: str | None = Field(
        default=None, description="Generated answer, or null if evidence is insufficient"
    )
    citations: list[Citation] = Field(default_factory=list)
    abstention_reason: str | None = Field(
        default=None,
        description="Reason for abstaining when answer is null",
    )
    synthesis_mode: str = Field(
        default="extractive",
        description="Synthesis method: 'extractive' or 'llm'",
    )
    claims: list[GroundedClaim] = Field(
        default_factory=list,
        description="Structured claims if synthesized by LLM",
    )


class VerificationResult(BaseModel):
    """Result of citation verification."""

    verified: bool = Field(..., description="Whether the quote was found in the source")
    match_type: str | None = Field(default=None, description="'exact' or 'normalized' if verified")
    match_position: int | None = Field(
        default=None, description="Character position of the match in source text"
    )
    detail: str = Field(default="", description="Human-readable verification detail")


class ExtractedPage(BaseModel):
    """A single page extracted from a PDF document."""

    page_number: int = Field(..., ge=1, description="1-indexed page number")
    text: str = Field(..., description="Extracted text for this page")
    word_count: int = Field(..., ge=0, description="Word count for page text")
    has_images: bool = Field(default=False, description="Whether page contains raster images")


class PDFExtractionResult(BaseModel):
    """Outcome of attempting to download and extract text from a PDF."""

    record_id: str
    pdf_url: str
    status: str = Field(
        ...,
        description=(
            "Status code: 'success', 'image_only', 'encrypted', 'empty', "
            "'corrupt', 'download_failed', 'unsupported'"
        ),
    )
    total_pages: int = Field(default=0, ge=0)
    extracted_pages: int = Field(default=0, ge=0)
    total_words: int = Field(default=0, ge=0)
    ocr_required: bool = Field(
        default=False, description="True when document is scanned or image-only"
    )
    error_message: str | None = Field(default=None)
    pages: list[ExtractedPage] = Field(default_factory=list)
    full_text: str = Field(default="", description="Combined extracted full text")


class BatchIngestResult(BaseModel):
    """Summary report from a bounded batch PDF ingestion run."""

    records_processed: int = 0
    pdfs_downloaded: int = 0
    pdfs_skipped: int = 0
    skip_reasons: dict[str, int] = Field(default_factory=dict)
    pages_extracted: int = 0
    passages_stored: int = 0
    extraction_failures: int = 0
    details: list[dict] = Field(default_factory=list)
