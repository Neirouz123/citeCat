"""Safe PDF downloading and full-text extraction for HAL publications."""

from __future__ import annotations

import asyncio
import email.utils
import hashlib
import ipaddress
import json
import logging
import os
import random
import re
import urllib.parse
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx

from citecat.config import Config
from citecat.models import ExtractedPage, PDFExtractionResult

logger = logging.getLogger(__name__)

# Lazy PyMuPDF import
try:
    import pymupdf  # type: ignore
except ImportError:
    try:
        import fitz as pymupdf  # type: ignore
    except ImportError:
        pymupdf = None


class PDFDownloadError(Exception):
    """Raised when PDF download fails permanently."""


class PDFSecurityError(PDFDownloadError):
    """Raised when URL scheme is forbidden or target violates SSRF policies."""


class PDFSizeLimitError(PDFDownloadError):
    """Raised when PDF file exceeds the configured maximum size limit."""


class PDFContentTypeError(PDFDownloadError):
    """Raised when HTTP response is not a valid PDF (e.g. HTML error page)."""


def is_safe_url(url: str, allow_private: bool = False) -> tuple[bool, str]:
    """Validate that a URL is safe to download from, enforcing strict SSRF protections.

    Checks:
    - Scheme must be 'http' or 'https' (rejects 'file:', 'ftp:', 'gopher:', etc.).
    - Hostname must be present and non-empty.
    - Prohibits 'localhost', '0.0.0.0', and internal domain suffixes.
    - Checks IP literals against loopback (127.0.0.0/8, ::1), private subnets,
      link-local/cloud metadata (169.254.0.0/16, fe80::/10),
      multicast, reserved, and unspecified addresses.

    Returns:
        A tuple of (is_safe, error_reason).
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string"

    try:
        parsed = urllib.parse.urlparse(url.strip())
    except Exception as exc:
        return False, f"Invalid URL syntax: {exc}"

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return (
            False,
            f"Prohibited URL scheme '{parsed.scheme}'; only 'http' and 'https' are allowed",
        )

    host = parsed.hostname
    if not host:
        return False, "URL lacks a valid hostname"

    host_lower = host.lower().strip(".")

    # Prohibit local/internal hostnames
    if host_lower in ("localhost", "0.0.0.0", "broadcasthost") or host_lower.endswith(
        (".local", ".internal", ".localhost", ".onion", ".test", ".example", ".invalid")
    ):
        return False, f"Access to internal or local hostname '{host}' is prohibited"

    if allow_private:
        return True, ""

    # Check if host is an IP literal
    ip_str = host_lower.strip("[]")
    try:
        ip = ipaddress.ip_address(ip_str)
        if ip.is_loopback:
            return False, f"Access to loopback IP address '{host}' is prohibited"
        if ip.is_link_local:
            return False, f"Access to link-local IP address '{host}' is prohibited"
        if ip.is_private:
            return False, f"Access to private IP address '{host}' is prohibited"
        if ip.is_multicast:
            return False, f"Access to multicast IP address '{host}' is prohibited"
        if ip.is_reserved:
            return False, f"Access to reserved IP address '{host}' is prohibited"
        if ip.is_unspecified:
            return False, f"Access to unspecified IP address '{host}' is prohibited"
    except ValueError:
        # Host is a domain name, not an IP literal
        pass

    return True, ""


class PDFDownloader:
    """Safe, rate-limited, SSRF-protected, and cached PDF downloader.

    Features:
    - Pre-request and redirect-hop SSRF validation (blocks localhost, private, and link-local IPs).
    - Manual redirect loop with strict hop limits.
    - Streaming download with maximum byte size enforcement.
    - Content-Type and PDF magic bytes (`%PDF-`) validation.
    - Respectful exponential backoff for HTTP 429 and 5xx errors with Retry-After support.
    - Atomic file system caching with SHA-256 verification and corruption recovery.
    """

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Config()
        self._client: httpx.AsyncClient | None = None
        self._cache_dir = (
            Path(self._config.pdf_cache_dir)
            if self._config.pdf_cache_dir
            else Path.cwd() / ".citecat_cache" / "pdfs"
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            # Note: follow_redirects=False because we manually track and validate every redirect hop
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._config.pdf_timeout),
                follow_redirects=False,
                headers={
                    "User-Agent": "CiteCat/0.1.0 (HAL Open-Access PDF Research Client)",
                    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
                },
                verify=self._config.hal_ssl_verify,
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _get_cache_paths(self, url: str) -> tuple[Path, Path]:
        """Return (pdf_path, meta_path) for a cached URL."""
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
        pdf_path = self._cache_dir / f"{url_hash}.pdf"
        meta_path = self._cache_dir / f"{url_hash}.meta.json"
        return pdf_path, meta_path

    def get_cached(self, url: str) -> bytes | None:
        """Return cached PDF bytes if available and valid; safely cleans up corrupted entries."""
        pdf_path, meta_path = self._get_cache_paths(url)
        if pdf_path.is_file() and meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                pdf_bytes = pdf_path.read_bytes()
                content_hash = hashlib.sha256(pdf_bytes).hexdigest()
                expected_hash = meta.get("sha256")
                expected_size = meta.get("size_bytes")
                if expected_hash == content_hash and (
                    expected_size is None or expected_size == len(pdf_bytes)
                ):
                    logger.debug("PDF cache hit for %s", url)
                    return pdf_bytes
                else:
                    logger.warning("Corrupt PDF cache entry for %s: hash/size mismatch", url)
            except Exception as exc:
                logger.warning("Corrupt PDF cache entry for %s: %s", url, exc)

            # Safe cleanup of corrupted cache files
            try:
                pdf_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
            except OSError:
                pass
        return None

    def _save_cache(self, url: str, content: bytes, content_type: str) -> None:
        """Atomically save downloaded PDF bytes and metadata to disk cache."""
        try:
            pdf_path, meta_path = self._get_cache_paths(url)
            tmp_id = f"{os.getpid()}_{uuid.uuid4().hex}"
            tmp_pdf = self._cache_dir / f".tmp_{tmp_id}.pdf"
            tmp_meta = self._cache_dir / f".tmp_{tmp_id}.meta.json"

            tmp_pdf.write_bytes(content)
            meta = {
                "url": url,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
                "content_type": content_type,
                "downloaded_at": datetime.now(UTC).isoformat(),
            }
            tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            # Atomic replace
            os.replace(tmp_pdf, pdf_path)
            os.replace(tmp_meta, meta_path)
        except Exception as exc:
            logger.warning("Failed to atomically write PDF cache for %s: %s", url, exc)
            try:
                if "tmp_pdf" in locals() and tmp_pdf.exists():
                    tmp_pdf.unlink(missing_ok=True)
                if "tmp_meta" in locals() and tmp_meta.exists():
                    tmp_meta.unlink(missing_ok=True)
            except OSError:
                pass

    def _parse_retry_after(self, retry_after: str | None, attempt: int) -> float:
        """Parse Retry-After header (seconds or RFC 9110 HTTP-date), with backoff fallback."""
        default_delay = min(
            self._config.retry_base_delay * (2**attempt) + random.uniform(0, 0.5),  # noqa: S311
            self._config.retry_max_delay,
        )
        if not retry_after:
            return default_delay

        retry_after = retry_after.strip()
        try:
            return max(0.0, min(float(retry_after), self._config.retry_max_delay))
        except ValueError:
            pass

        try:
            parsed_date = email.utils.parsedate_to_datetime(retry_after)
            delay = (parsed_date - datetime.now(UTC)).total_seconds()
            return max(0.0, min(delay, self._config.retry_max_delay))
        except Exception:
            return default_delay

    async def download(self, url: str) -> bytes:
        """Download a PDF safely with SSRF checks, streaming size limits, and validation.

        Args:
            url: The HTTP(S) URL of the PDF document.

        Returns:
            The raw bytes of the downloaded PDF.

        Raises:
            PDFSecurityError: If URL scheme is forbidden or target violates SSRF policies.
            PDFSizeLimitError: If the file exceeds `config.pdf_max_size_bytes`.
            PDFContentTypeError: If the response is not a PDF (e.g. HTML error page).
            PDFDownloadError: For network or HTTP status errors.
        """
        # 1. SSRF Pre-validation
        safe, reason = is_safe_url(url, allow_private=self._config.pdf_allow_private_ips)
        if not safe:
            raise PDFSecurityError(f"Rejected unsafe PDF URL '{url}': {reason}")

        # 2. Check disk cache
        cached = self.get_cached(url)
        if cached is not None:
            return cached

        client = await self._get_client()
        max_size = self._config.pdf_max_size_bytes
        max_retries = self._config.max_retries
        max_redirects = self._config.pdf_max_redirects

        current_url = url
        redirect_count = 0

        while True:
            # Validate URL before every connection / redirect hop
            safe, reason = is_safe_url(
                current_url, allow_private=self._config.pdf_allow_private_ips
            )
            if not safe:
                raise PDFSecurityError(f"Rejected unsafe redirect target '{current_url}': {reason}")

            attempt = 0
            while attempt <= max_retries:
                try:
                    async with client.stream("GET", current_url) as resp:
                        # 3. Handle HTTP Redirects manually with SSRF checks
                        if resp.status_code in (301, 302, 303, 307, 308):
                            redirect_count += 1
                            if redirect_count > max_redirects:
                                raise PDFDownloadError(
                                    f"Exceeded maximum redirect limit of {max_redirects} for {url}"
                                )
                            location = resp.headers.get("Location")
                            if not location:
                                raise PDFDownloadError(
                                    f"HTTP redirect {resp.status_code} missing Location header "
                                    f"for {current_url}"
                                )
                            target_url = urllib.parse.urljoin(current_url, location)
                            current_url = target_url
                            # Break inner retry loop to request redirect target
                            break

                        # 4. Handle HTTP Errors & Retryable Status Codes
                        if resp.status_code >= 400:
                            if resp.status_code == 429 or resp.status_code >= 500:
                                if attempt < max_retries:
                                    retry_after_str = resp.headers.get("Retry-After")
                                    delay = self._parse_retry_after(retry_after_str, attempt)
                                    logger.warning(
                                        "PDF download %s returned %d (attempt %d/%d), "
                                        "retrying in %.1fs",
                                        current_url,
                                        resp.status_code,
                                        attempt + 1,
                                        max_retries + 1,
                                        delay,
                                    )
                                    await asyncio.sleep(delay)
                                    attempt += 1
                                    continue
                                raise PDFDownloadError(
                                    f"HTTP {resp.status_code} error downloading PDF from "
                                    f"{current_url} (retries exhausted)"
                                )
                            raise PDFDownloadError(
                                f"HTTP {resp.status_code} non-retryable error downloading PDF from "
                                f"{current_url}"
                            )

                        # 5. Validate Content-Type
                        ct = resp.headers.get("Content-Type", "").lower()
                        if "text/html" in ct or "text/plain" in ct:
                            raise PDFContentTypeError(
                                f"Expected PDF but server returned Content-Type '{ct}' for "
                                f"{current_url}"
                            )

                        # 6. Check Content-Length if present
                        cl_header = resp.headers.get("Content-Length")
                        if cl_header:
                            try:
                                content_length = int(cl_header)
                                if content_length > max_size:
                                    raise PDFSizeLimitError(
                                        f"PDF size {content_length} bytes exceeds maximum limit "
                                        f"of {max_size} bytes ({max_size // (1024 * 1024)} MB)"
                                    )
                            except ValueError:
                                pass

                        # 7. Stream chunks with running byte count enforcement
                        chunks: list[bytes] = []
                        total_bytes = 0
                        async for chunk in resp.aiter_bytes(chunk_size=65536):
                            total_bytes += len(chunk)
                            if total_bytes > max_size:
                                raise PDFSizeLimitError(
                                    f"PDF download exceeded maximum size limit of {max_size} bytes"
                                )
                            chunks.append(chunk)

                        content = b"".join(chunks)

                        # 8. Magic byte validation: %PDF- in first 1024 bytes
                        if b"%PDF-" not in content[:1024]:
                            raise PDFContentTypeError(
                                f"File from {current_url} lacks valid '%PDF-' header magic bytes"
                            )

                        # 9. Atomic cache save and return
                        self._save_cache(url, content, ct)
                        return content

                except (PDFSecurityError, PDFSizeLimitError, PDFContentTypeError):
                    raise
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    if attempt < max_retries:
                        delay = min(
                            self._config.retry_base_delay * (2**attempt) + random.uniform(0, 0.5),  # noqa: S311
                            self._config.retry_max_delay,
                        )
                        logger.warning(
                            "PDF network error (%s) on attempt %d/%d, retrying in %.1fs",
                            exc,
                            attempt + 1,
                            max_retries + 1,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        attempt += 1
                    else:
                        raise PDFDownloadError(
                            f"Network error downloading PDF from {current_url}: {exc}"
                        ) from exc
                except Exception as exc:
                    raise PDFDownloadError(
                        f"Failed to download PDF from {current_url}: {exc}"
                    ) from exc


class PDFExtractor:
    """Extracts text page-by-page from PDF documents using PyMuPDF."""

    @staticmethod
    def is_available() -> bool:
        """Return True if PyMuPDF (or fitz) is installed and available."""
        return pymupdf is not None

    @classmethod
    def extract_text(
        cls,
        pdf_bytes: bytes,
        record_id: str,
        pdf_url: str,
        max_pages: int = 1000,
        max_size_bytes: int = 50 * 1024 * 1024,
    ) -> PDFExtractionResult:
        """Extract text page by page from raw PDF bytes.

        Args:
            pdf_bytes: Raw bytes of the PDF document.
            record_id: The HAL document identifier.
            pdf_url: The source PDF URL.
            max_pages: Maximum number of pages to process (guards against page decompression bombs).
            max_size_bytes: Maximum byte size permitted for processing.

        Returns:
            A populated :class:`PDFExtractionResult`.
        """
        if not cls.is_available():
            return PDFExtractionResult(
                record_id=record_id,
                pdf_url=pdf_url,
                status="unsupported",
                error_message=(
                    "PDF extraction library (pymupdf) is not installed. "
                    "Install optional dependencies with: uv pip install 'citecat[pdf]'"
                ),
            )

        if not pdf_bytes:
            return PDFExtractionResult(
                record_id=record_id,
                pdf_url=pdf_url,
                status="empty",
                error_message="Provided PDF byte stream is empty.",
            )

        if len(pdf_bytes) > max_size_bytes:
            return PDFExtractionResult(
                record_id=record_id,
                pdf_url=pdf_url,
                status="size_limit",
                error_message=(
                    f"PDF byte size {len(pdf_bytes)} exceeds extraction limit "
                    f"of {max_size_bytes} bytes."
                ),
            )

        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            return PDFExtractionResult(
                record_id=record_id,
                pdf_url=pdf_url,
                status="corrupt",
                error_message=f"Failed to parse PDF document: {exc}",
            )

        try:
            if doc.is_encrypted:
                return PDFExtractionResult(
                    record_id=record_id,
                    pdf_url=pdf_url,
                    status="encrypted",
                    total_pages=len(doc),
                    error_message="Document is password encrypted and cannot be extracted.",
                )

            total_pages = len(doc)
            if total_pages == 0:
                return PDFExtractionResult(
                    record_id=record_id,
                    pdf_url=pdf_url,
                    status="empty",
                    total_pages=0,
                    error_message="PDF contains 0 pages.",
                )

            pages_to_extract = min(total_pages, max_pages)
            pages: list[ExtractedPage] = []
            has_any_images = False
            total_words = 0

            for page_idx in range(pages_to_extract):
                page_num = page_idx + 1
                page = doc[page_idx]

                # Check for images on page
                try:
                    images = page.get_images()
                    page_has_images = len(images) > 0
                except Exception:
                    page_has_images = False

                if page_has_images:
                    has_any_images = True

                # Extract text
                raw_text = page.get_text("text") or ""
                norm_text = cls._normalize_page_text(raw_text)
                words = re.findall(r"\S+", norm_text)
                word_count = len(words)
                total_words += word_count

                pages.append(
                    ExtractedPage(
                        page_number=page_num,
                        text=norm_text,
                        word_count=word_count,
                        has_images=page_has_images,
                    )
                )

            # Combined full text: stitch pages separated by a clean marker
            # We use '\n\n' between pages so exact offsets map cleanly
            full_text = "\n\n".join(p.text for p in pages if p.text.strip())

            # Detect image-only / scanned PDFs where OCR is required
            avg_words = total_words / pages_to_extract if pages_to_extract > 0 else 0
            if (total_words < 30 or avg_words < 10) and has_any_images and pages_to_extract > 0:
                return PDFExtractionResult(
                    record_id=record_id,
                    pdf_url=pdf_url,
                    status="image_only",
                    total_pages=total_pages,
                    extracted_pages=len(pages),
                    total_words=total_words,
                    ocr_required=True,
                    error_message=(
                        "Document appears to be scanned or image-only. "
                        "Full text extraction requires OCR (not currently enabled)."
                    ),
                    pages=pages,
                    full_text=full_text,
                )

            if total_words == 0:
                return PDFExtractionResult(
                    record_id=record_id,
                    pdf_url=pdf_url,
                    status="empty",
                    total_pages=total_pages,
                    extracted_pages=len(pages),
                    total_words=0,
                    ocr_required=False,
                    error_message="PDF document contains no extractable text.",
                    pages=pages,
                    full_text="",
                )

            return PDFExtractionResult(
                record_id=record_id,
                pdf_url=pdf_url,
                status="success",
                total_pages=total_pages,
                extracted_pages=len(pages),
                total_words=total_words,
                ocr_required=False,
                pages=pages,
                full_text=full_text,
            )

        finally:
            doc.close()

    @staticmethod
    def _normalize_page_text(text: str) -> str:
        """Conservatively normalize whitespace without corrupting verbatim quotations."""
        if not text:
            return ""
        # Normalize non-breaking spaces and line separators
        text = text.replace("\u00a0", " ").replace("\u202f", " ")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Strip trailing whitespace on each line
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        # Rejoin lines with single newlines, collapsing 3+ consecutive newlines to 2
        joined = "\n".join(lines)
        joined = re.sub(r"\n{3,}", "\n\n", joined)
        return joined.strip()
