"""Asynchronous HAL API client with retries, rate limiting, and concurrency control."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from citecat.config import Config
from citecat.models import Record

logger = logging.getLogger(__name__)

# Fields we request from the HAL Solr API
_HAL_FIELDS = (
    "docid,halId_s,title_s,abstract_s,authFullName_s,"
    "uri_s,licence_s,language_s,producedDateY_i,producedDate_s,"
    "fileMain_s,files_s,fileLicenses_s,fileType_s,openAccess_bool"
)


class HALClientError(Exception):
    """Raised for non-retryable HAL API errors."""


class HALClient:
    """Async client for the HAL open-science search API.

    Features:
    - Configurable timeout and concurrency via semaphore.
    - Respectful rate limiting (token bucket).
    - Exponential backoff with jitter for HTTP 429 and 5xx errors.
    - Honors ``Retry-After`` headers.
    """

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Config()
        self._client: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(self._config.hal_max_concurrent)
        self._rate_limit_lock = asyncio.Lock()
        self._next_request_time: float = 0.0
        self._min_interval: float = 1.0 / self._config.rate_limit_per_second

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._config.hal_base_url,
                timeout=httpx.Timeout(self._config.hal_timeout),
                headers={"Accept": "application/json"},
                verify=self._config.hal_ssl_verify,
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # -- Rate limiting --------------------------------------------------

    async def _wait_for_rate_limit(self) -> None:
        async with self._rate_limit_lock:
            now = time.monotonic()
            target_time = max(now, self._next_request_time)
            self._next_request_time = target_time + self._min_interval
            delay = target_time - now
        if delay > 0:
            await asyncio.sleep(delay)

    # -- Retry logic ----------------------------------------------------

    async def _request_with_retry(
        self, method: str, url: str, params: dict[str, Any] | None = None
    ) -> httpx.Response:
        """Make an HTTP request with retry for transient failures."""
        last_exc: Exception | None = None
        for attempt in range(self._config.max_retries + 1):
            async with self._semaphore:
                await self._wait_for_rate_limit()
                try:
                    client = await self._get_client()
                    resp = await client.request(method, url, params=params)

                    if resp.status_code < 400:
                        return resp

                    # Permanent client errors (except 429) – don't retry
                    if 400 <= resp.status_code < 500 and resp.status_code != 429:
                        raise HALClientError(
                            f"HAL API returned {resp.status_code}: {resp.text[:500]}"
                        )

                    # Retryable: 429 or 5xx
                    delay = self._backoff_delay(attempt, resp)
                    logger.warning(
                        "HAL API %s (attempt %d/%d), retrying in %.1fs",
                        resp.status_code,
                        attempt + 1,
                        self._config.max_retries + 1,
                        delay,
                    )
                    last_exc = HALClientError(f"HAL API returned {resp.status_code}")

                except httpx.TimeoutException as exc:
                    delay = self._backoff_delay(attempt, None)
                    logger.warning(
                        "HAL API timeout (attempt %d/%d), retrying in %.1fs",
                        attempt + 1,
                        self._config.max_retries + 1,
                        delay,
                    )
                    last_exc = exc

                except httpx.HTTPError as exc:
                    delay = self._backoff_delay(attempt, None)
                    logger.warning(
                        "HAL API network error (attempt %d/%d): %s",
                        attempt + 1,
                        self._config.max_retries + 1,
                        exc,
                    )
                    last_exc = exc

            if attempt < self._config.max_retries:
                await asyncio.sleep(delay)

        msg = f"HAL API request failed after {self._config.max_retries + 1} attempts"
        raise HALClientError(msg) from last_exc

    def _backoff_delay(self, attempt: int, response: httpx.Response | None) -> float:
        """Calculate delay with exponential backoff, jitter, and Retry-After."""
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    delay_val = float(retry_after)
                    return max(0.0, min(delay_val, self._config.retry_max_delay))
                except ValueError:
                    try:
                        import email.utils

                        dt = email.utils.parsedate_to_datetime(retry_after)
                        delay_val = (dt - datetime.now(UTC)).total_seconds()
                        return max(0.0, min(delay_val, self._config.retry_max_delay))
                    except Exception:
                        pass

        base = self._config.retry_base_delay * (2**attempt)
        jitter = random.uniform(0, base * 0.5)  # noqa: S311
        return min(base + jitter, self._config.retry_max_delay)

    # -- Public API -----------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, str] | None = None,
        start: int = 0,
    ) -> list[Record]:
        """Search HAL for publications matching *query*.

        Args:
            query: Free-text search query.
            limit: Maximum number of results (capped at ``hal_max_results``).
            filters: Optional Solr filter-query dict, e.g. ``{"docType_s": "ART"}``.
            start: Pagination offset index.

        Returns:
            List of validated :class:`Record` objects.
        """
        if not query or not query.strip():
            return []
        limit = max(1, min(limit, self._config.hal_max_results))
        params: dict[str, Any] = {
            "q": query,
            "wt": "json",
            "fl": _HAL_FIELDS,
            "rows": limit,
            "start": max(0, start),
        }
        if filters:
            fq_parts = [f"{k}:{v}" for k, v in filters.items()]
            params["fq"] = fq_parts

        resp = await self._request_with_retry("GET", "/search/", params=params)
        return self._parse_search_response(resp)

    async def fetch(self, record_id: str) -> Record | None:
        """Fetch a single HAL record by its identifier.

        Args:
            record_id: HAL identifier (e.g. ``"hal-01234567"``) or docid.

        Returns:
            A validated :class:`Record` or ``None`` if not found.
        """
        if not record_id or not record_id.strip():
            return None
        # Support both halId_s and docid identifiers safely
        safe_id = record_id.replace('"', "")
        params: dict[str, Any] = {
            "q": f'halId_s:"{safe_id}" OR docid:"{safe_id}"',
            "wt": "json",
            "fl": _HAL_FIELDS,
            "rows": 1,
        }
        resp = await self._request_with_retry("GET", "/search/", params=params)
        records = self._parse_search_response(resp)
        return records[0] if records else None

    # -- Response parsing -----------------------------------------------

    def _parse_search_response(self, resp: httpx.Response) -> list[Record]:
        """Convert a HAL JSON response into validated Records."""
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Failed to parse HAL response JSON: %s", exc)
            return []

        docs = data.get("response", {}).get("docs", [])
        records: list[Record] = []
        now = datetime.now(UTC)

        for doc in docs:
            record = self._doc_to_record(doc, now)
            if record is not None:
                records.append(record)

        return records

    @staticmethod
    def _doc_to_record(doc: dict[str, Any], retrieved_at: datetime) -> Record | None:
        """Convert a single HAL Solr document to a Record.

        Returns ``None`` if required fields are missing (source_url is
        mandatory for citation provenance).
        """
        # HAL identifier
        hal_id = doc.get("halId_s")
        if not hal_id:
            docid = doc.get("docid")
            if docid:
                hal_id = str(docid)
            else:
                return None

        # Title – may be a list
        title_raw = doc.get("title_s")
        if isinstance(title_raw, list):
            title = title_raw[0] if title_raw else ""
        else:
            title = title_raw or ""
        if not title:
            return None

        # Source URL – mandatory for provenance
        source_url = doc.get("uri_s", "")
        if not source_url:
            return None

        # License – represent unknown explicitly
        license_raw = doc.get("licence_s")
        if isinstance(license_raw, list):
            license_val = license_raw[0] if license_raw else "unknown"
        else:
            license_val = license_raw or "unknown"
        if not isinstance(license_val, str) or not license_val.strip():
            license_val = "unknown"

        # Abstract / snippet
        abstract = doc.get("abstract_s")
        if isinstance(abstract, list):
            snippet = abstract[0] if abstract else ""
        else:
            snippet = abstract or ""

        # Authors
        authors = doc.get("authFullName_s", [])
        if isinstance(authors, str):
            authors = [authors]

        # Language
        language_raw = doc.get("language_s")
        language = None
        if isinstance(language_raw, list) and language_raw:
            language = language_raw[0]
        elif isinstance(language_raw, str):
            language = language_raw

        # Publication date
        pub_date = doc.get("producedDate_s") or None
        if pub_date is None:
            year = doc.get("producedDateY_i")
            if year is not None:
                pub_date = str(year)

        # Open-access PDF and deposited file metadata
        files_raw = doc.get("files_s", [])
        if isinstance(files_raw, str):
            files_list = [files_raw]
        elif isinstance(files_raw, list):
            files_list = [str(f) for f in files_raw if f]
        else:
            files_list = []

        # Determine primary PDF URL
        pdf_url: str | None = None
        file_main = doc.get("fileMain_s")
        if isinstance(file_main, str) and file_main.strip():
            pdf_url = file_main.strip()
        elif isinstance(file_main, list) and file_main:
            pdf_url = str(file_main[0]).strip()

        if not pdf_url:
            for f in files_list:
                if f.lower().endswith(".pdf") or "/file/" in f.lower():
                    pdf_url = f
                    break

        if not pdf_url:
            # Check legacy/alternative fields if present
            for fb_key in ("fileDepoire_s", "fileAnnexes_s"):
                fb_val = doc.get(fb_key)
                if isinstance(fb_val, list) and fb_val:
                    pdf_url = str(fb_val[0]).strip()
                    break
                elif isinstance(fb_val, str) and fb_val.strip():
                    pdf_url = fb_val.strip()
                    break

        # License for deposited files
        pdf_license: str | None = None
        file_lic_raw = doc.get("fileLicenses_s")
        if isinstance(file_lic_raw, list) and file_lic_raw:
            pdf_license = str(file_lic_raw[0]).strip() if file_lic_raw[0] else None
        elif isinstance(file_lic_raw, str) and file_lic_raw.strip():
            pdf_license = file_lic_raw.strip()

        open_access = bool(doc.get("openAccess_bool", False))

        try:
            return Record(
                id=hal_id,
                title=title,
                snippet=snippet,
                source_url=source_url,
                license=license_val,
                retrieved_at=retrieved_at,
                authors=authors,
                language=language,
                publication_date=pub_date,
                pdf_url=pdf_url,
                pdf_license=pdf_license,
                files=files_list,
                open_access=open_access,
            )
        except Exception as exc:
            logger.debug("Skipping invalid HAL doc %s: %s", hal_id, exc)
            return None


def make_cache_key(query: str, limit: int, filters: dict[str, str] | None) -> str:
    """Produce a stable cache key for a search request."""
    payload = json.dumps(
        {"q": query, "limit": limit, "filters": filters or {}},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()
