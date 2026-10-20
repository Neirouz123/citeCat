"""Tests for citecat.client – HAL API client with mocked HTTP."""

from __future__ import annotations

import httpx
import pytest
import respx

from citecat.client import HALClient, HALClientError, make_cache_key
from citecat.config import Config


@pytest.fixture
def config():
    return Config(
        hal_base_url="https://api.archives-ouvertes.fr",
        hal_timeout=5.0,
        hal_max_concurrent=2,
        hal_max_results=50,
        max_retries=2,
        retry_base_delay=0.01,
        retry_max_delay=0.1,
        rate_limit_per_second=100.0,  # fast for tests
    )


@pytest.fixture
def client(config):
    return HALClient(config)


def _hal_response(docs: list[dict], num_found: int | None = None) -> dict:
    """Build a minimal HAL-style Solr response."""
    return {
        "response": {
            "numFound": num_found or len(docs),
            "start": 0,
            "docs": docs,
        }
    }


def _sample_doc(**overrides) -> dict:
    defaults = {
        "docid": 12345,
        "halId_s": "hal-00012345",
        "title_s": ["A Sample Publication"],
        "abstract_s": ["This is the abstract."],
        "authFullName_s": ["Alice Smith", "Bob Jones"],
        "uri_s": "https://hal.science/hal-00012345",
        "licence_s": "cc-by",
        "language_s": ["en"],
        "producedDateY_i": 2023,
    }
    defaults.update(overrides)
    return defaults


class TestSearchParsing:
    """Test HAL response conversion to validated Records."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_returns_records(self, client):
        respx.get("https://api.archives-ouvertes.fr/search/").mock(
            return_value=httpx.Response(200, json=_hal_response([_sample_doc()]))
        )
        records = await client.search("test query")
        assert len(records) == 1
        r = records[0]
        assert r.id == "hal-00012345"
        assert r.title == "A Sample Publication"
        assert r.source_url == "https://hal.science/hal-00012345"
        assert r.license == "cc-by"
        assert len(r.authors) == 2
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_empty_query(self, client):
        records = await client.search("")
        assert records == []
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_empty_results(self, client):
        respx.get("https://api.archives-ouvertes.fr/search/").mock(
            return_value=httpx.Response(200, json=_hal_response([]))
        )
        records = await client.search("nonexistent topic xyz")
        assert records == []
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_missing_uri_skips_record(self, client):
        doc = _sample_doc(uri_s="")
        respx.get("https://api.archives-ouvertes.fr/search/").mock(
            return_value=httpx.Response(200, json=_hal_response([doc]))
        )
        records = await client.search("test")
        assert len(records) == 0
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_missing_license_defaults_unknown(self, client):
        doc = _sample_doc(licence_s=None)
        respx.get("https://api.archives-ouvertes.fr/search/").mock(
            return_value=httpx.Response(200, json=_hal_response([doc]))
        )
        records = await client.search("test")
        assert len(records) == 1
        assert records[0].license == "unknown"
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_malformed_json(self, client):
        respx.get("https://api.archives-ouvertes.fr/search/").mock(
            return_value=httpx.Response(200, text="not valid json!!!")
        )
        records = await client.search("test")
        assert records == []
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_title_as_string(self, client):
        doc = _sample_doc(title_s="Plain Title")
        respx.get("https://api.archives-ouvertes.fr/search/").mock(
            return_value=httpx.Response(200, json=_hal_response([doc]))
        )
        records = await client.search("test")
        assert records[0].title == "Plain Title"
        await client.close()


class TestFetch:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_found(self, client):
        respx.get("https://api.archives-ouvertes.fr/search/").mock(
            return_value=httpx.Response(200, json=_hal_response([_sample_doc()]))
        )
        record = await client.fetch("hal-00012345")
        assert record is not None
        assert record.id == "hal-00012345"
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_not_found(self, client):
        respx.get("https://api.archives-ouvertes.fr/search/").mock(
            return_value=httpx.Response(200, json=_hal_response([]))
        )
        record = await client.fetch("hal-99999999")
        assert record is None
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_empty_id(self, client):
        record = await client.fetch("")
        assert record is None
        await client.close()


class TestRetries:
    @respx.mock
    @pytest.mark.asyncio
    async def test_retry_on_500(self, client):
        route = respx.get("https://api.archives-ouvertes.fr/search/")
        route.side_effect = [
            httpx.Response(500, text="Internal Server Error"),
            httpx.Response(500, text="Internal Server Error"),
            httpx.Response(200, json=_hal_response([_sample_doc()])),
        ]
        records = await client.search("test")
        assert len(records) == 1
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_retry_on_429(self, client):
        route = respx.get("https://api.archives-ouvertes.fr/search/")
        route.side_effect = [
            httpx.Response(429, text="Too Many Requests", headers={"Retry-After": "0.01"}),
            httpx.Response(200, json=_hal_response([_sample_doc()])),
        ]
        records = await client.search("test")
        assert len(records) == 1
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_retry_on_400(self, client):
        respx.get("https://api.archives-ouvertes.fr/search/").mock(
            return_value=httpx.Response(400, text="Bad Request")
        )
        with pytest.raises(HALClientError, match="400"):
            await client.search("test")
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_exhausted_retries(self, client):
        respx.get("https://api.archives-ouvertes.fr/search/").mock(
            return_value=httpx.Response(500, text="error")
        )
        with pytest.raises(HALClientError, match="failed after"):
            await client.search("test")
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, client):
        route = respx.get("https://api.archives-ouvertes.fr/search/")
        route.side_effect = [
            httpx.ConnectTimeout("timeout"),
            httpx.Response(200, json=_hal_response([_sample_doc()])),
        ]
        records = await client.search("test")
        assert len(records) == 1
        await client.close()


class TestCacheKey:
    def test_stable_key(self):
        k1 = make_cache_key("test", 10, None)
        k2 = make_cache_key("test", 10, None)
        assert k1 == k2

    def test_different_queries(self):
        k1 = make_cache_key("test1", 10, None)
        k2 = make_cache_key("test2", 10, None)
        assert k1 != k2

    def test_different_limits(self):
        k1 = make_cache_key("test", 10, None)
        k2 = make_cache_key("test", 20, None)
        assert k1 != k2

    def test_filters_affect_key(self):
        k1 = make_cache_key("test", 10, None)
        k2 = make_cache_key("test", 10, {"docType_s": "ART"})
        assert k1 != k2


class TestAdvancedClientFeatures:
    @respx.mock
    @pytest.mark.asyncio
    async def test_search_with_filters(self, client):
        route = respx.get("https://api.archives-ouvertes.fr/search/").mock(
            return_value=httpx.Response(200, json=_hal_response([_sample_doc()]))
        )
        records = await client.search("climate", filters={"docType_s": "ART"})
        assert len(records) == 1
        assert route.called
        req_params = route.calls.last.request.url.params
        assert "fq" in req_params
        assert req_params["fq"] == "docType_s:ART"
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_with_pagination(self, client):
        route = respx.get("https://api.archives-ouvertes.fr/search/").mock(
            return_value=httpx.Response(200, json=_hal_response([_sample_doc()]))
        )
        records = await client.search("climate", limit=5, start=10)
        assert len(records) == 1
        req_params = route.calls.last.request.url.params
        assert req_params["start"] == "10"
        assert req_params["rows"] == "5"
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_retry_on_429_http_date(self, client):
        route = respx.get("https://api.archives-ouvertes.fr/search/")
        route.side_effect = [
            httpx.Response(
                429,
                text="Rate limited",
                headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"},
            ),
            httpx.Response(200, json=_hal_response([_sample_doc()])),
        ]
        records = await client.search("test")
        assert len(records) == 1
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_licence_parsed(self, client):
        doc = _sample_doc(licence_s=["https://creativecommons.org/licenses/by/4.0/"])
        respx.get("https://api.archives-ouvertes.fr/search/").mock(
            return_value=httpx.Response(200, json=_hal_response([doc]))
        )
        records = await client.search("test")
        assert len(records) == 1
        assert records[0].license == "https://creativecommons.org/licenses/by/4.0/"
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_concurrent_requests_handled(self, client):
        import asyncio

        respx.get("https://api.archives-ouvertes.fr/search/").mock(
            return_value=httpx.Response(200, json=_hal_response([_sample_doc()]))
        )
        # Execute 5 concurrent search requests
        tasks = [client.search(f"query_{i}") for i in range(5)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 5
        for res in results:
            assert len(res) == 1
        await client.close()
