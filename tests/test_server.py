"""Tests for citecat.server – MCP tool registration and execution."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest
import respx

from citecat.config import Config
from citecat.models import Record
from citecat.server import (
    _get_store,
    mcp_server,
)


@pytest.fixture(autouse=True)
def clean_environment(tmp_path, monkeypatch):
    """Isolate DBs for each server test."""
    db_cache = str(tmp_path / "srv_cache.db")
    db_retrieval = str(tmp_path / "srv_retrieval.db")
    monkeypatch.setenv("CITECAT_CACHE_DB_PATH", db_cache)
    monkeypatch.setenv("CITECAT_RETRIEVAL_DB_PATH", db_retrieval)

    # Reset server singletons
    import citecat.server as srv

    srv._config = None
    srv._hal_client = None
    srv._pdf_downloader = None
    srv._cache = None
    srv._store = None
    srv._retriever = None
    yield


class TestMCPRegistration:
    """Verify that tools and resource templates are properly registered on MCPServer."""

    def test_server_name(self):
        assert mcp_server.name == "citecat"

    @pytest.mark.asyncio
    async def test_tools_registered(self):
        tools = await mcp_server.list_tools()
        names = {t.name for t in tools}
        assert "search" in names
        assert "get_record" in names
        assert "answer_with_citations" in names
        assert "verify_citation" in names
        assert "ingest_pdf" in names

    @pytest.mark.asyncio
    async def test_resource_template_registered(self):
        templates = await mcp_server.list_resource_templates()
        assert any(t.uri_template == "catalogue://record/{id}" for t in templates)


class TestMCPExecution:
    """Verify calling tools and reading resources through MCPServer API."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_search_tool_execution(self):
        sample_doc = {
            "docid": 111,
            "halId_s": "hal-00000111",
            "title_s": ["Quantum Computing Basics"],
            "abstract_s": ["An introduction to quantum gates."],
            "authFullName_s": ["Alice Q."],
            "uri_s": "https://hal.science/hal-00000111",
            "licence_s": "cc-by",
            "language_s": ["en"],
            "producedDateY_i": 2024,
        }
        respx.get("https://api.archives-ouvertes.fr/search/").mock(
            return_value=httpx.Response(
                200,
                json={"response": {"numFound": 1, "start": 0, "docs": [sample_doc]}},
            )
        )
        res = await mcp_server.call_tool("search", {"query": "quantum computing", "limit": 5})
        assert not res.is_error
        data = json.loads(res.content[0].text)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["id"] == "hal-00000111"
        assert data[0]["license"] == "cc-by"

    @pytest.mark.asyncio
    async def test_search_tool_empty_query(self):
        res = await mcp_server.call_tool("search", {"query": "   "})
        assert not res.is_error
        data = json.loads(res.content[0].text)
        assert "error" in data

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_record_tool_execution(self):
        sample_doc = {
            "docid": 222,
            "halId_s": "hal-00000222",
            "title_s": ["Neural Network Architecture"],
            "abstract_s": ["Transformer models are scalable."],
            "authFullName_s": ["Bob AI"],
            "uri_s": "https://hal.science/hal-00000222",
            "licence_s": "unknown",
            "language_s": ["en"],
            "producedDateY_i": 2023,
        }
        respx.get("https://api.archives-ouvertes.fr/search/").mock(
            return_value=httpx.Response(
                200,
                json={"response": {"numFound": 1, "start": 0, "docs": [sample_doc]}},
            )
        )
        res = await mcp_server.call_tool("get_record", {"id": "hal-00000222"})
        assert not res.is_error
        data = json.loads(res.content[0].text)
        assert data["id"] == "hal-00000222"
        assert data["title"] == "Neural Network Architecture"

    @pytest.mark.asyncio
    async def test_answer_with_citations_and_abstention(self):
        store = _get_store()
        record = Record(
            id="hal-100",
            title="Vaccine Efficacy Study",
            snippet=(
                "The clinical trial demonstrated that the mRNA vaccine produced a "
                "high neutralizing antibody response in participants, with an estimated "
                "efficacy rate of 94 percent against severe symptoms."
            ),
            source_url="https://hal.science/hal-100",
            license="cc-by-4.0",
            retrieved_at=datetime.now(UTC),
        )
        store.ingest_record(record)

        # 1. Answerable question
        res = await mcp_server.call_tool(
            "answer_with_citations",
            {"question": "vaccine efficacy neutralizing antibody", "k": 3},
        )
        assert not res.is_error
        data = json.loads(res.content[0].text)
        assert data["answer"] is not None
        assert len(data["citations"]) >= 1
        assert data["citations"][0]["source_url"] == "https://hal.science/hal-100"
        assert data["citations"][0]["license"] == "cc-by-4.0"

        # 2. Unanswerable question -> abstention
        res_abs = await mcp_server.call_tool(
            "answer_with_citations",
            {"question": "ancient Roman gladiator weapons", "k": 3},
        )
        assert not res_abs.is_error
        data_abs = json.loads(res_abs.content[0].text)
        assert data_abs["answer"] is None
        assert data_abs["abstention_reason"] is not None

    @pytest.mark.asyncio
    async def test_verify_citation_tool_execution(self):
        store = _get_store()
        text = (
            "Open science mandates the dissemination of research publications "
            "and research data free of charge."
        )
        record = Record(
            id="hal-300",
            title="Open Science Policy",
            snippet=text,
            source_url="https://hal.science/hal-300",
            license="cc-by",
            retrieved_at=datetime.now(UTC),
        )
        store.ingest_record(record, full_text=text)

        # Exact quote verification
        res = await mcp_server.call_tool(
            "verify_citation",
            {
                "quote": "dissemination of research publications",
                "source_url": "https://hal.science/hal-300",
            },
        )
        data = json.loads(res.content[0].text)
        assert data["verified"] is True
        assert data["match_type"] == "exact"
        assert data["match_position"] > 0

        # Normalized quote verification
        res_norm = await mcp_server.call_tool(
            "verify_citation",
            {
                "quote": "DISSEMINATION  OF  RESEARCH  PUBLICATIONS",
                "source_url": "https://hal.science/hal-300",
            },
        )
        data_norm = json.loads(res_norm.content[0].text)
        assert data_norm["verified"] is True
        assert data_norm["match_type"] == "normalized"

        # Fabricated quote verification failure
        res_fab = await mcp_server.call_tool(
            "verify_citation",
            {
                "quote": "Open science requires paying 500 dollars per paper",
                "source_url": "https://hal.science/hal-300",
            },
        )
        data_fab = json.loads(res_fab.content[0].text)
        assert data_fab["verified"] is False

    @pytest.mark.asyncio
    async def test_read_catalogue_resource(self):
        store = _get_store()
        record = Record(
            id="hal-400",
            title="Resource Catalog Test",
            snippet="Catalog snippet text.",
            source_url="https://hal.science/hal-400",
            license="mit",
            retrieved_at=datetime.now(UTC),
        )
        store.ingest_record(record, full_text="Full catalog text here.")

        contents = await mcp_server.read_resource("catalogue://record/hal-400")
        assert len(contents) >= 1
        data = json.loads(contents[0].content)
        assert data["id"] == "hal-400"
        assert data["source_url"] == "https://hal.science/hal-400"
        assert data["license"] == "mit"


class TestMCPConfig:
    def test_config_from_env(self):
        config = Config.from_env()
        assert config.hal_base_url == "https://api.archives-ouvertes.fr"
        assert config.hal_timeout == 30.0
        assert config.cache_ttl_seconds == 3600
