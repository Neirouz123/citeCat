"""Comprehensive tests for PDF downloading, extraction, provenance, and ingestion."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest
import respx

from citecat.citations import build_answer_with_citations, verify_citation
from citecat.config import Config
from citecat.ingest import IngestStore
from citecat.models import Record
from citecat.pdf import (
    PDFContentTypeError,
    PDFDownloader,
    PDFDownloadError,
    PDFExtractor,
    PDFSecurityError,
    PDFSizeLimitError,
    is_safe_url,
)
from citecat.server import ingest_batch_records, ingest_pdf


def _create_synthetic_pdf(
    pages_text: list[str],
    with_image: bool = False,
    password: str | None = None,
) -> bytes:
    """Helper to generate in-memory synthetic PDF bytes using PyMuPDF."""
    import pymupdf

    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text)
        if with_image:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 50, 50), False)
            pix.clear_with(200)
            page.insert_image(pymupdf.Rect(50, 150, 100, 200), pixmap=pix)

    if password:
        # Encryption with user password
        perm = int(
            pymupdf.PDF_PERM_ACCESSIBILITY
            | pymupdf.PDF_PERM_PRINT
            | pymupdf.PDF_PERM_COPY
            | pymupdf.PDF_PERM_ANNOTATE
        )
        pdf_bytes = doc.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw=password,
            permissions=perm,
        )
    else:
        pdf_bytes = doc.tobytes()

    doc.close()
    return pdf_bytes


# ===========================================================================
# 1. Extraction Unit Tests
# ===========================================================================
class TestPDFExtraction:
    """Tests for PDFExtractor text extraction and page parsing."""

    def test_single_page_extraction(self):
        content = "This is a single page scientific paper on computational astrophysics."
        pdf_bytes = _create_synthetic_pdf([content])

        res = PDFExtractor.extract_text(pdf_bytes, "hal-01", "https://hal.science/doc.pdf")
        assert res.status == "success"
        assert res.total_pages == 1
        assert res.extracted_pages == 1
        assert res.ocr_required is False
        assert "computational astrophysics" in res.full_text
        assert len(res.pages) == 1
        assert res.pages[0].page_number == 1
        assert res.pages[0].word_count >= 8

    def test_multi_page_extraction_preserves_page_numbers(self):
        page1 = "Page 1: Introduction to quantum error correction codes."
        page2 = "Page 2: Fault-tolerant threshold theorem and surface code lattices."
        page3 = "Page 3: Experimental implementation on superconducting transmon qubits."
        pdf_bytes = _create_synthetic_pdf([page1, page2, page3])

        res = PDFExtractor.extract_text(pdf_bytes, "hal-02", "https://hal.science/doc.pdf")
        assert res.status == "success"
        assert res.total_pages == 3
        assert res.extracted_pages == 3
        assert len(res.pages) == 3
        assert [p.page_number for p in res.pages] == [1, 2, 3]
        assert "superconducting transmon qubits" in res.pages[2].text

    def test_empty_pdf_detection(self):
        pdf_bytes = _create_synthetic_pdf([""])
        res = PDFExtractor.extract_text(pdf_bytes, "hal-empty", "https://hal.science/empty.pdf")
        assert res.status in ("empty", "image_only")

    def test_empty_bytes_handling(self):
        res = PDFExtractor.extract_text(b"", "hal-empty-bytes", "https://hal.science/empty.pdf")
        assert res.status == "empty"
        assert "empty" in res.error_message.lower()

    def test_corrupt_pdf_handling(self):
        res = PDFExtractor.extract_text(
            b"%PDF-1.4 completely corrupt random garbage", "hal-bad", "https://hal.science/bad.pdf"
        )
        assert res.status == "corrupt"
        assert res.error_message is not None

    def test_scanned_image_only_pdf_detects_ocr_required(self):
        # Empty text + image on page
        pdf_bytes = _create_synthetic_pdf([""], with_image=True)
        res = PDFExtractor.extract_text(pdf_bytes, "hal-scanned", "https://hal.science/scan.pdf")
        assert res.status == "image_only"
        assert res.ocr_required is True
        assert "ocr" in res.error_message.lower()

    def test_encrypted_pdf_detection(self):
        pdf_bytes = _create_synthetic_pdf(["Secret text"], password="secretpassword")
        res = PDFExtractor.extract_text(pdf_bytes, "hal-locked", "https://hal.science/locked.pdf")
        assert res.status == "encrypted"
        assert "encrypted" in res.error_message.lower()

    def test_whitespace_normalization_preserves_accents_and_quotes(self):
        raw = "Étude approfondie : l'acidification des océans « altère » les récifs coralliens.\n\n"
        norm = PDFExtractor._normalize_page_text(raw)
        assert "Étude approfondie" in norm
        assert "l'acidification" in norm
        assert "« altère »" in norm


# ===========================================================================
# 2. Downloader Safety and Validation Tests
# ===========================================================================
class TestPDFDownloader:
    """Tests for safe PDF downloading, timeouts, size limits, and caching."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_successful_download_and_caching(self, tmp_path):
        config = Config(pdf_cache_dir=str(tmp_path / "pdf_cache"))
        downloader = PDFDownloader(config)

        pdf_bytes = _create_synthetic_pdf(["Research paper content."])
        url = "https://hal.science/test/paper.pdf"
        respx.get(url).respond(
            200,
            headers={"Content-Type": "application/pdf", "Content-Length": str(len(pdf_bytes))},
            content=pdf_bytes,
        )

        try:
            downloaded = await downloader.download(url)
            assert downloaded == pdf_bytes

            # Second call must hit disk cache without making network request
            respx.get(url).respond(500)  # If network called, would fail
            cached = await downloader.download(url)
            assert cached == pdf_bytes
        finally:
            await downloader.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_size_limit_via_header_rejected(self, tmp_path):
        # Max 1 KB limit
        config = Config(pdf_cache_dir=str(tmp_path), pdf_max_size_bytes=1024)
        downloader = PDFDownloader(config)

        url = "https://hal.science/giant.pdf"
        respx.get(url).respond(
            200,
            headers={"Content-Type": "application/pdf", "Content-Length": "2048000"},
            content=b"%PDF-1.4 " + b"0" * 2000,
        )

        try:
            with pytest.raises(PDFSizeLimitError) as exc_info:
                await downloader.download(url)
            assert "exceeds maximum limit" in str(exc_info.value)
        finally:
            await downloader.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_size_limit_via_streaming_chunks_rejected(self, tmp_path):
        # 500 byte limit without Content-Length header
        config = Config(pdf_cache_dir=str(tmp_path), pdf_max_size_bytes=500)
        downloader = PDFDownloader(config)

        url = "https://hal.science/stream.pdf"
        respx.get(url).respond(
            200,
            headers={"Content-Type": "application/pdf"},
            content=b"%PDF-1.4 " + b"A" * 2000,
        )

        try:
            with pytest.raises(PDFSizeLimitError):
                await downloader.download(url)
        finally:
            await downloader.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_html_error_page_rejected(self, tmp_path):
        config = Config(pdf_cache_dir=str(tmp_path))
        downloader = PDFDownloader(config)

        url = "https://hal.science/login.html"
        respx.get(url).respond(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            content=b"<html><body>Please login to institutional repository</body></html>",
        )

        try:
            with pytest.raises(PDFContentTypeError) as exc_info:
                await downloader.download(url)
            assert "Content-Type" in str(exc_info.value)
        finally:
            await downloader.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_invalid_magic_bytes_rejected(self, tmp_path):
        config = Config(pdf_cache_dir=str(tmp_path))
        downloader = PDFDownloader(config)

        url = "https://hal.science/fake.pdf"
        respx.get(url).respond(
            200,
            headers={"Content-Type": "application/pdf"},
            content=b"NOT_A_PDF_DOCUMENT_BINARY",
        )

        try:
            with pytest.raises(PDFContentTypeError) as exc_info:
                await downloader.download(url)
            assert "%PDF-" in str(exc_info.value)
        finally:
            await downloader.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_http_404_error_handling(self, tmp_path):
        config = Config(pdf_cache_dir=str(tmp_path))
        downloader = PDFDownloader(config)

        url = "https://hal.science/notfound.pdf"
        respx.get(url).respond(404)

        try:
            with pytest.raises(PDFDownloadError) as exc_info:
                await downloader.download(url)
            assert "404" in str(exc_info.value)
        finally:
            await downloader.close()


# ===========================================================================
# 3. Store Ingestion and Citation Provenance Tests
# ===========================================================================
class TestPDFIngestionAndProvenance:
    """Tests for SQLite persistence, offsets, page numbers, and citation verification."""

    def test_ingest_pdf_offsets_and_provenance(self, tmp_path):
        config = Config(retrieval_db_path=str(tmp_path / "test_pdf_store.db"))
        store = IngestStore(config)

        p1_text = (
            "Section 1: Atmospheric Carbon Dioxide Dynamics and Biogeochemical Feedbacks.\n"
            "Anthropogenic carbon dioxide emissions accelerate ocean acidification and alter "
            "terrestrial vegetative biomass sinks across continental biomes."
        )
        p2_text = (
            "Section 3: Cryospheric Observations and Glacial Retreat Rates.\n"
            "Longitudinal satellite radar interferometry demonstrates that sub-polar ice shelves "
            "are thinning at an annualized velocity of 3.4 meters per annum."
        )
        pdf_bytes = _create_synthetic_pdf([p1_text, p2_text])
        pdf_url = "https://hal.science/hal-09999999/file/climate.pdf"
        extraction = PDFExtractor.extract_text(pdf_bytes, "hal-09999999", pdf_url)

        rec = Record(
            id="hal-09999999",
            title="Atmospheric Carbon and Cryospheric Thinning",
            snippet="Abstract snippet.",
            source_url="https://hal.science/hal-09999999",
            license="https://creativecommons.org/licenses/by/4.0/",
            pdf_url=pdf_url,
            pdf_license="https://creativecommons.org/licenses/by/4.0/",
            retrieved_at=datetime.now(UTC),
        )

        try:
            passages_created = store.ingest_pdf(rec, extraction)
            assert passages_created == 2

            passages = store.get_all_passages()
            assert len(passages) == 2

            full_text = store.get_record_text("hal-09999999")
            assert full_text is not None

            # Verify exact offsets map to stored full_text
            for p in passages:
                assert p.record_id == "hal-09999999"
                assert p.source_type == "pdf"
                assert p.pdf_url == pdf_url
                assert p.page_number in (1, 2)
                # Exact character offset guarantee
                assert full_text[p.start_char : p.end_char] == p.text

            # Verify citation construction and verification
            ans = build_answer_with_citations(
                "satellite radar interferometry glacial retreat rate", store, min_score=0.1
            )
            assert ans.answer is not None
            assert len(ans.citations) >= 1
            top_cit = ans.citations[0]
            assert top_cit.record_id == "hal-09999999"
            assert top_cit.source_type == "pdf"
            assert top_cit.page_number == 2
            assert top_cit.pdf_url == pdf_url

            # Citation verification against full text
            v_res = verify_citation(top_cit.quote, full_text)
            assert v_res.verified is True
            assert v_res.match_type == "exact"

            # Check that answer text contains Page 2 indicator
            assert "Page 2" in ans.answer

        finally:
            store.close()

    def test_reingest_pdf_purges_old_passages_and_embeddings(self, tmp_path):
        config = Config(retrieval_db_path=str(tmp_path / "reingest.db"))
        store = IngestStore(config)

        rec = Record(
            id="hal-reingest-01",
            title="Initial Title",
            snippet="Initial abstract snippet",
            source_url="https://hal.science/hal-reingest-01",
            license="CC-BY",
            retrieved_at=datetime.now(UTC),
        )

        try:
            # 1. Ingest as abstract
            store.ingest_record(rec)
            assert len(store.get_all_passages()) == 1
            meta = store.get_record_metadata("hal-reingest-01")
            assert meta["source_type"] == "abstract"

            # Store an embedding
            store.store_embedding("hal-reingest-01", 0, 10, "Initial ab", "test-model", [0.1] * 128)
            assert store.count_embeddings() == 1

            # 2. Ingest full-text PDF for same record
            pdf_bytes = _create_synthetic_pdf(["Updated full text page 1", "Updated page 2"])
            extraction = PDFExtractor.extract_text(
                pdf_bytes, "hal-reingest-01", "https://hal.science/updated.pdf"
            )
            store.ingest_pdf(rec, extraction)

            # Old passages and embeddings must be cleanly purged
            passages = store.get_all_passages()
            assert len(passages) == 2
            assert all(p.source_type == "pdf" for p in passages)
            assert store.count_embeddings() == 0  # Invalidated stale embeddings
            meta = store.get_record_metadata("hal-reingest-01")
            assert meta["source_type"] == "pdf"
        finally:
            store.close()

    def test_abstention_on_unanswerable_query(self, tmp_path):
        config = Config(retrieval_db_path=str(tmp_path / "abstain.db"))
        store = IngestStore(config)

        pdf_bytes = _create_synthetic_pdf(["Quantum entanglement in trapped atomic ions."])
        extraction = PDFExtractor.extract_text(
            pdf_bytes, "hal-quantum", "https://hal.science/q.pdf"
        )
        rec = Record(
            id="hal-quantum",
            title="Quantum Entanglement",
            snippet="Snippet",
            source_url="https://hal.science/q",
            license="CC-BY",
            retrieved_at=datetime.now(UTC),
        )
        try:
            store.ingest_pdf(rec, extraction)
            # Irrelevant query must strictly abstain
            ans = build_answer_with_citations(
                "How to bake French chocolate cake", store, min_score=0.35
            )
            assert ans.answer is None
            assert ans.abstention_reason is not None
            assert len(ans.citations) == 0
        finally:
            store.close()


# ===========================================================================
# 4. MCP Tool and Batch Ingestion Tests
# ===========================================================================
class TestMCPPDFIntegration:
    """Tests for MCP ingest_pdf tool and bounded batch ingestion."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_mcp_ingest_pdf_tool(self, tmp_path, monkeypatch):
        # Configure test databases
        monkeypatch.setenv("CITECAT_RETRIEVAL_DB_PATH", str(tmp_path / "mcp_retrieval.db"))
        monkeypatch.setenv("CITECAT_CACHE_DB_PATH", str(tmp_path / "mcp_cache.db"))
        monkeypatch.setenv("CITECAT_PDF_CACHE_DIR", str(tmp_path / "mcp_pdfs"))

        pdf_bytes = _create_synthetic_pdf(["Full text of scientific article on genomics."])
        hal_id = "hal-05555555"
        pdf_url = "https://hal.science/hal-05555555/document"

        # Mock HAL Solr search response
        solr_resp = {
            "response": {
                "numFound": 1,
                "docs": [
                    {
                        "halId_s": hal_id,
                        "title_s": "Genomic Advances",
                        "uri_s": f"https://hal.science/{hal_id}",
                        "licence_s": "https://creativecommons.org/licenses/by/4.0/",
                        "fileMain_s": pdf_url,
                        "openAccess_bool": True,
                    }
                ],
            }
        }
        respx.get("https://api.archives-ouvertes.fr/search/").respond(200, json=solr_resp)
        respx.get(pdf_url).respond(
            200,
            headers={"Content-Type": "application/pdf", "Content-Length": str(len(pdf_bytes))},
            content=pdf_bytes,
        )

        res_str = await ingest_pdf(hal_id)
        data = json.loads(res_str)

        assert data["status"] == "success"
        assert data["record_id"] == hal_id
        assert data["total_pages"] == 1
        assert data["passages_stored"] >= 1
        assert data["ocr_required"] is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_mcp_ingest_pdf_no_file_available(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CITECAT_RETRIEVAL_DB_PATH", str(tmp_path / "mcp_retrieval.db"))

        hal_id = "hal-no-file-01"
        solr_resp = {
            "response": {
                "numFound": 1,
                "docs": [
                    {
                        "halId_s": hal_id,
                        "title_s": "Conference Abstract Without Deposit",
                        "uri_s": f"https://hal.science/{hal_id}",
                        "licence_s": "unknown",
                        "openAccess_bool": False,
                    }
                ],
            }
        }
        respx.get("https://api.archives-ouvertes.fr/search/").respond(200, json=solr_resp)

        res_str = await ingest_pdf(hal_id)
        data = json.loads(res_str)

        assert data["status"] == "no_pdf_available"
        assert data["passages_stored"] == 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_ingest_batch_records_bounded(self, tmp_path):
        config = Config(
            retrieval_db_path=str(tmp_path / "batch.db"),
            pdf_cache_dir=str(tmp_path / "batch_pdfs"),
        )
        pdf_bytes = _create_synthetic_pdf(["Batch document content."])
        pdf_url = "https://hal.science/doc1.pdf"

        solr_resp = {
            "response": {
                "numFound": 2,
                "docs": [
                    {
                        "halId_s": "hal-batch-1",
                        "title_s": "Batch Record 1",
                        "uri_s": "https://hal.science/hal-batch-1",
                        "licence_s": "CC-BY",
                        "fileMain_s": pdf_url,
                        "openAccess_bool": True,
                    },
                    {
                        "halId_s": "hal-batch-2",
                        "title_s": "Batch Record 2 Without File",
                        "uri_s": "https://hal.science/hal-batch-2",
                        "licence_s": "CC-BY",
                        "openAccess_bool": False,
                    },
                ],
            }
        }
        respx.get("https://api.archives-ouvertes.fr/search/").respond(200, json=solr_resp)
        respx.get(pdf_url).respond(
            200,
            headers={"Content-Type": "application/pdf"},
            content=pdf_bytes,
        )

        result = await ingest_batch_records(query="genomics", limit=2, config=config)

        assert result.records_processed == 2
        assert result.pdfs_downloaded == 1
        assert result.pdfs_skipped == 1
        assert result.skip_reasons.get("no_pdf_url") == 1
        assert result.pages_extracted == 1
        assert result.passages_stored >= 1
        assert result.extraction_failures == 0


# ===========================================================================
# 5. Security & SSRF Boundary Tests
# ===========================================================================
class TestPDFSecurityAndHardening:
    """Rigorous offline security tests for URL schemes, SSRF targets, and provenance."""

    def test_is_safe_url_scheme_validation(self):
        """Verify prohibited schemes are rejected."""
        assert is_safe_url("file:///etc/passwd")[0] is False
        assert is_safe_url("ftp://example.com/doc.pdf")[0] is False
        assert is_safe_url("gopher://example.com")[0] is False
        assert is_safe_url("javascript:alert(1)")[0] is False
        assert is_safe_url("data:application/pdf;base64,xxxx")[0] is False
        assert is_safe_url("")[0] is False

        # Legitimate HTTP and HTTPS URLs are accepted
        assert is_safe_url("https://hal.science/hal-01234567/document")[0] is True
        assert is_safe_url("http://api.archives-ouvertes.fr/doc.pdf")[0] is True

    def test_is_safe_url_ssrf_rejection(self):
        """Verify localhost, private IPs, link-local metadata, and internal domains are rejected."""
        # Loopback
        assert is_safe_url("http://localhost/doc.pdf")[0] is False
        assert is_safe_url("http://127.0.0.1/doc.pdf")[0] is False
        assert is_safe_url("http://127.0.0.2/doc.pdf")[0] is False
        assert is_safe_url("http://[::1]/doc.pdf")[0] is False

        # Private IPv4 subnets
        assert is_safe_url("http://10.0.0.1/doc.pdf")[0] is False
        assert is_safe_url("http://172.16.0.1/doc.pdf")[0] is False
        assert is_safe_url("http://192.168.1.1/doc.pdf")[0] is False

        # Link-local / Cloud Metadata (AWS/GCP/Azure IMDS)
        assert is_safe_url("http://169.254.169.254/latest/meta-data/")[0] is False
        assert is_safe_url("http://[fe80::1]/doc.pdf")[0] is False

        # Unspecified / Broadcast
        assert is_safe_url("http://0.0.0.0/doc.pdf")[0] is False

        # Internal/local hostnames
        assert is_safe_url("http://myinternal.local/doc.pdf")[0] is False
        assert is_safe_url("http://corp.internal/doc.pdf")[0] is False
        assert is_safe_url("http://service.localhost/doc.pdf")[0] is False

    @pytest.mark.asyncio
    async def test_download_rejects_ssrf_without_network_request(self, tmp_path):
        """Downloader must block unsafe URLs immediately before establishing any connection."""
        config = Config(pdf_cache_dir=str(tmp_path / "pdf_cache"))
        downloader = PDFDownloader(config)

        try:
            with pytest.raises(PDFSecurityError, match="Rejected unsafe PDF URL"):
                await downloader.download("http://127.0.0.1:8000/doc.pdf")

            with pytest.raises(PDFSecurityError, match="Rejected unsafe PDF URL"):
                await downloader.download("file:///etc/passwd")

            with pytest.raises(PDFSecurityError, match="link-local IP"):
                await downloader.download("http://169.254.169.254/latest/meta-data/")

            with pytest.raises(PDFSecurityError, match="private IP"):
                await downloader.download("http://10.0.0.1/doc.pdf")
        finally:
            await downloader.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_redirect_to_private_ip_is_blocked(self, tmp_path):
        """Redirect from an apparently safe public URL to an internal IP must be blocked."""
        config = Config(pdf_cache_dir=str(tmp_path / "pdf_cache"), pdf_max_redirects=5)
        downloader = PDFDownloader(config)

        # Mock public URL redirecting to private cloud metadata service
        respx.get("https://hal.science/safe-redirect.pdf").respond(
            302, headers={"Location": "http://169.254.169.254/latest/meta-data"}
        )

        try:
            with pytest.raises(PDFSecurityError, match="Rejected unsafe redirect target"):
                await downloader.download("https://hal.science/safe-redirect.pdf")
        finally:
            await downloader.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_redirect_limit_enforcement(self, tmp_path):
        """Exceeding configured max_redirects must terminate with PDFDownloadError."""
        config = Config(pdf_cache_dir=str(tmp_path / "pdf_cache"), pdf_max_redirects=2)
        downloader = PDFDownloader(config)

        respx.get("https://hal.science/r1").respond(
            302, headers={"Location": "https://hal.science/r2"}
        )
        respx.get("https://hal.science/r2").respond(
            302, headers={"Location": "https://hal.science/r3"}
        )
        respx.get("https://hal.science/r3").respond(
            302, headers={"Location": "https://hal.science/r4"}
        )

        try:
            with pytest.raises(PDFDownloadError, match="Exceeded maximum redirect limit"):
                await downloader.download("https://hal.science/r1")
        finally:
            await downloader.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_size_limit_without_content_length(self, tmp_path):
        """Streaming size limit must trigger even if server omits Content-Length header."""
        config = Config(pdf_cache_dir=str(tmp_path / "pdf_cache"), pdf_max_size_bytes=500)
        downloader = PDFDownloader(config)

        # 2000 bytes streamed with no Content-Length header
        big_content = b"%PDF-1.4\n" + b"X" * 2000
        respx.get("https://hal.science/oversized.pdf").respond(
            200,
            headers={"Content-Type": "application/pdf"},
            stream=httpx.ByteStream(big_content),
        )

        try:
            with pytest.raises(PDFSizeLimitError, match="exceeded maximum size limit"):
                await downloader.download("https://hal.science/oversized.pdf")
        finally:
            await downloader.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_corrupt_cache_recovery_and_cleanup(self, tmp_path):
        """Corrupt cache entry must be detected, cleared, and refetched safely."""
        config = Config(pdf_cache_dir=str(tmp_path / "pdf_cache"))
        downloader = PDFDownloader(config)

        url = "https://hal.science/corrupt-test.pdf"
        valid_pdf = _create_synthetic_pdf(["Valid document content."])

        # 1. First fetch creates valid cache
        respx.get(url).respond(
            200,
            headers={"Content-Type": "application/pdf"},
            content=valid_pdf,
        )
        fetched = await downloader.download(url)
        assert fetched == valid_pdf

        # 2. Corrupt the cached PDF file on disk
        pdf_path, meta_path = downloader._get_cache_paths(url)
        assert pdf_path.is_file() and meta_path.is_file()
        pdf_path.write_bytes(b"tampered corrupt content")

        # 3. Reading corrupt cache should detect hash mismatch, clean up files, and return None
        cached = downloader.get_cached(url)
        assert cached is None
        # Corrupt files must have been unlinked
        assert not pdf_path.exists()
        assert not meta_path.exists()

        await downloader.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_atomic_cache_write_leaves_no_tmp_files(self, tmp_path):
        """Atomic cache save must not leave temporary files behind."""
        cache_dir = tmp_path / "atomic_cache"
        config = Config(pdf_cache_dir=str(cache_dir))
        downloader = PDFDownloader(config)

        url = "https://hal.science/atomic-test.pdf"
        pdf_bytes = _create_synthetic_pdf(["Atomic test."])
        respx.get(url).respond(
            200,
            headers={"Content-Type": "application/pdf"},
            content=pdf_bytes,
        )

        try:
            await downloader.download(url)
            # Verify no temporary files exist
            tmp_files = list(cache_dir.glob(".tmp_*"))
            assert len(tmp_files) == 0

            # Verify valid cache files exist
            pdf_path, meta_path = downloader._get_cache_paths(url)
            assert pdf_path.is_file()
            assert meta_path.is_file()
        finally:
            await downloader.close()

    def test_extractor_bounded_by_max_pages(self):
        """PDFExtractor must respect max_pages limit to guard against decompression bombs."""
        pages = [f"Page {i} content text." for i in range(1, 11)]
        pdf_bytes = _create_synthetic_pdf(pages)

        res = PDFExtractor.extract_text(
            pdf_bytes,
            record_id="hal-bomb",
            pdf_url="https://hal.science/bomb.pdf",
            max_pages=3,
        )

        assert res.total_pages == 10
        assert res.extracted_pages == 3
        assert len(res.pages) == 3
        assert [p.page_number for p in res.pages] == [1, 2, 3]

    def test_extractor_bounded_by_max_size(self):
        """PDFExtractor must reject inputs exceeding max_size_bytes."""
        pdf_bytes = _create_synthetic_pdf(["Some content."])
        res = PDFExtractor.extract_text(
            pdf_bytes,
            record_id="hal-large",
            pdf_url="https://hal.science/large.pdf",
            max_size_bytes=10,  # lower than synthetic PDF size
        )
        assert res.status == "size_limit"
        assert "exceeds extraction limit" in res.error_message

    def test_ingest_pdf_defense_on_non_success_extraction(self, tmp_path):
        """IngestStore.ingest_pdf must skip ingestion when extraction status != success."""
        config = Config(retrieval_db_path=str(tmp_path / "defense.db"))
        store = IngestStore(config)

        rec = Record(
            id="hal-def",
            title="Defense Record",
            snippet="Abstract snippet",
            source_url="https://hal.science/def",
            license="CC-BY",
            retrieved_at=datetime.now(UTC),
        )
        # Store initial abstract
        store.ingest_record(rec)

        # Call ingest_pdf with corrupt extraction
        from citecat.models import PDFExtractionResult

        corrupt_result = PDFExtractionResult(
            record_id="hal-def",
            pdf_url="https://hal.science/def.pdf",
            status="corrupt",
            error_message="Corrupt PDF",
        )
        count = store.ingest_pdf(rec, corrupt_result)
        assert count == 0

        # Verify original abstract was not erased
        assert store.get_record_text("hal-def") == "Abstract snippet"
        store.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_mcp_ingest_pdf_security_violation_response(self, tmp_path, monkeypatch):
        """MCP ingest_pdf must return security_violation when PDF URL is an internal target."""
        monkeypatch.setenv("CITECAT_RETRIEVAL_DB_PATH", str(tmp_path / "mcp_sec.db"))
        monkeypatch.setenv("CITECAT_CACHE_DB_PATH", str(tmp_path / "mcp_sec_cache.db"))

        hal_id = "hal-sec-1"
        solr_resp = {
            "response": {
                "numFound": 1,
                "docs": [
                    {
                        "halId_s": hal_id,
                        "title_s": "SSRF Target Paper",
                        "uri_s": f"https://hal.science/{hal_id}",
                        "licence_s": "CC-BY",
                        "fileMain_s": "http://127.0.0.1/internal_secret.pdf",
                        "openAccess_bool": True,
                    }
                ],
            }
        }
        respx.get("https://api.archives-ouvertes.fr/search/").respond(200, json=solr_resp)

        res_str = await ingest_pdf(hal_id)
        data = json.loads(res_str)

        assert data["status"] == "security_violation"
        assert "Security violation" in data["error_message"]
        assert data["passages_stored"] == 0
