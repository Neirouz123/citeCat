"""Tests for citecat.models – Pydantic validation."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from citecat.models import (
    AnswerResult,
    Citation,
    Passage,
    Record,
    VerificationResult,
)


class TestRecord:
    """Record model validation tests."""

    def _make_record(self, **overrides):
        defaults = {
            "id": "hal-00000001",
            "title": "Test Publication",
            "snippet": "An abstract.",
            "source_url": "https://hal.science/hal-00000001",
            "license": "cc-by",
            "retrieved_at": datetime(2024, 1, 1, tzinfo=UTC),
        }
        defaults.update(overrides)
        return Record(**defaults)

    def test_valid_record(self):
        r = self._make_record()
        assert r.id == "hal-00000001"
        assert r.license == "cc-by"

    def test_missing_source_url_raises(self):
        with pytest.raises(ValidationError):
            self._make_record(source_url="")

    def test_whitespace_source_url_raises(self):
        with pytest.raises(ValidationError):
            self._make_record(source_url="   ")

    def test_missing_license_raises(self):
        with pytest.raises(ValidationError):
            self._make_record(license="")

    def test_whitespace_license_raises(self):
        with pytest.raises(ValidationError):
            self._make_record(license="   ")

    def test_missing_title_raises(self):
        with pytest.raises(ValidationError):
            self._make_record(title="")

    def test_unknown_license_allowed(self):
        r = self._make_record(license="unknown")
        assert r.license == "unknown"

    def test_optional_fields_default(self):
        r = self._make_record()
        assert r.authors == []
        assert r.language is None
        assert r.publication_date is None

    def test_optional_fields_set(self):
        r = self._make_record(
            authors=["Alice", "Bob"],
            language="en",
            publication_date="2024",
        )
        assert len(r.authors) == 2
        assert r.language == "en"

    def test_snippet_defaults_to_empty(self):
        r = self._make_record(snippet="")
        assert r.snippet == ""


class TestPassage:
    def test_valid_passage(self):
        p = Passage(
            record_id="hal-1",
            text="Some text here.",
            start_char=0,
            end_char=15,
            word_count=3,
        )
        assert p.word_count == 3

    def test_negative_start_raises(self):
        with pytest.raises(ValidationError):
            Passage(
                record_id="hal-1",
                text="x",
                start_char=-1,
                end_char=1,
                word_count=1,
            )


class TestCitation:
    def test_valid_citation(self):
        c = Citation(
            quote="some text",
            source_url="https://hal.science/hal-1",
            license="cc-by",
            record_id="hal-1",
            score=1.5,
        )
        assert c.score == 1.5


class TestAnswerResult:
    def test_abstention(self):
        a = AnswerResult(answer=None, abstention_reason="Not enough evidence.")
        assert a.answer is None
        assert a.abstention_reason is not None

    def test_with_answer(self):
        a = AnswerResult(
            answer="Here is the answer.",
            citations=[
                Citation(
                    quote="q",
                    source_url="u",
                    license="l",
                    record_id="r",
                    score=1.0,
                )
            ],
        )
        assert len(a.citations) == 1


class TestVerificationResult:
    def test_verified(self):
        v = VerificationResult(
            verified=True,
            match_type="exact",
            match_position=42,
            detail="ok",
        )
        assert v.verified

    def test_not_verified(self):
        v = VerificationResult(verified=False, detail="not found")
        assert not v.verified
