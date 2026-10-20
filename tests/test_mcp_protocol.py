"""End-to-end MCP protocol integration tests using the official MCP ClientSession over stdio.

Tests run completely offline and deterministically by pointing the child MCP
server subprocess to an ephemeral local mock HTTP server that returns HAL JSON
responses.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# Canned HAL records for deterministic offline protocol testing
SAMPLE_DOC = {
    "docid": "1234567",
    "halId_s": "hal-01234567",
    "uri_s": "https://hal.science/hal-01234567",
    "title_s": ["Quantum Computing Fundamentals and Algorithms"],
    "abstract_s": [
        "Quantum computing utilizes quantum superposition and quantum entanglement "
        "to perform complex calculations exponentially faster than classical computers "
        "for specific mathematical problem classes and cryptographic analyses."
    ],
    "authFullName_s": ["Alice Curie", "Bob Turing"],
    "language_s": ["en"],
    "producedDateY_i": 2024,
    "licence_s": "https://creativecommons.org/licenses/by/4.0/",
}

HAL_SEARCH_RESPONSE = {
    "response": {
        "numFound": 1,
        "start": 0,
        "docs": [SAMPLE_DOC],
    }
}

HAL_EMPTY_RESPONSE = {
    "response": {
        "numFound": 0,
        "start": 0,
        "docs": [],
    }
}


class MockHalHandler(BaseHTTPRequestHandler):
    """Minimal mock HTTP handler emulating HAL API search and fetch."""

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        params = parse_qs(parsed_url.query)
        q = params.get("q", [""])[0]

        if "error_500" in q:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Internal Server Error"}')
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        if "quantum" in q.lower() or 'halId_s:"hal-01234567"' in q:
            payload = HAL_SEARCH_RESPONSE
        else:
            payload = HAL_EMPTY_RESPONSE

        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:
        # Silence HTTP server logging during tests
        pass


@pytest.fixture(scope="module")
def mock_hal_server():
    """Start an ephemeral HTTP server on 127.0.0.1 with port 0."""
    server = HTTPServer(("127.0.0.1", 0), MockHalHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()


@pytest.fixture
def mcp_server_params(mock_hal_server, tmp_path):
    """Construct StdioServerParameters pointing to citecat.server with isolated DBs."""
    src_dir = str(Path(__file__).resolve().parent.parent / "src")

    # Inherit required OS environment variables for Windows/cross-platform subprocesses
    env = {
        "CITECAT_HAL_BASE_URL": mock_hal_server,
        "CITECAT_CACHE_DB_PATH": str(tmp_path / "protocol_cache.db"),
        "CITECAT_RETRIEVAL_DB_PATH": str(tmp_path / "protocol_retrieval.db"),
        "PYTHONPATH": src_dir,
    }
    for key in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "COMSPEC", "PATHEXT", "PYTHONHOME"):
        val = os.environ.get(key)
        if val is not None:
            env[key] = val

    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "citecat.server"],
        env=env,
    )


@pytest.mark.asyncio
async def test_mcp_handshake_and_capabilities(mcp_server_params):
    """Verify MCP protocol initialization handshake and server capabilities."""
    async with (
        stdio_client(mcp_server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        init_result = await session.initialize()

        # Verify protocol version and server metadata
        assert init_result.protocol_version is not None
        assert init_result.server_info.name == "citecat"
        assert init_result.server_info.version == "0.1.0"

        # Verify registered tools
        tools_result = await session.list_tools()
        tool_names = {t.name for t in tools_result.tools}
        assert "search" in tool_names
        assert "get_record" in tool_names
        assert "answer_with_citations" in tool_names
        assert "verify_citation" in tool_names
        assert "ingest_pdf" in tool_names

        for tool in tools_result.tools:
            assert tool.description, f"Tool '{tool.name}' must have a description"
            assert "Inputs:" in tool.description or "Limitations:" in tool.description

        # Verify registered resource templates
        templates_result = await session.list_resource_templates()
        template_uris = {t.uri_template for t in templates_result.resource_templates}
        assert "catalogue://record/{id}" in template_uris


@pytest.mark.asyncio
async def test_mcp_complete_workflow(mcp_server_params):
    """Exercise complete CiteCat workflow through MCP ClientSession over stdio.

    Workflow:
    1. Search for records via `search` tool.
    2. Inspect record metadata and source URLs.
    3. Retrieve the record via `get_record` tool.
    4. Access the record via `catalogue://record/{id}` resource.
    5. Ask an answerable question via `answer_with_citations`.
    6. Verify the returned citations via `verify_citation`.
    7. Ask an unanswerable question and verify abstention.
    """
    async with (
        stdio_client(mcp_server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        # 1. Search tool
        search_res = await session.call_tool("search", {"query": "quantum computing", "limit": 2})
        assert not search_res.is_error
        assert len(search_res.content) >= 1
        records = json.loads(search_res.content[0].text)
        assert isinstance(records, list)
        assert len(records) == 1

        rec = records[0]
        assert rec["id"] == "hal-01234567"
        assert "Quantum" in rec["title"]
        assert rec["source_url"] == "https://hal.science/hal-01234567"
        assert "creativecommons.org" in rec["license"]

        # 2. Get record tool
        fetch_res = await session.call_tool("get_record", {"id": "hal-01234567"})
        assert not fetch_res.is_error
        fetched = json.loads(fetch_res.content[0].text)
        assert fetched["id"] == "hal-01234567"
        assert fetched["source_url"] == "https://hal.science/hal-01234567"

        # 3. Read resource
        resource_res = await session.read_resource("catalogue://record/hal-01234567")
        assert len(resource_res.contents) >= 1
        res_data = json.loads(resource_res.contents[0].text)
        assert res_data["id"] == "hal-01234567"
        assert "full_text_preview" in res_data
        assert "Quantum computing" in res_data["full_text_preview"]

        # 4. Answer with citations (answerable question)
        qa_res = await session.call_tool(
            "answer_with_citations",
            {"question": "How does quantum computing perform calculations?", "k": 5},
        )
        assert not qa_res.is_error
        qa_data = json.loads(qa_res.content[0].text)
        assert qa_data["answer"] is not None
        assert len(qa_data["citations"]) >= 1

        citation = qa_data["citations"][0]
        assert citation["record_id"] == "hal-01234567"
        assert citation["source_url"] == "https://hal.science/hal-01234567"
        assert citation["license"] == "https://creativecommons.org/licenses/by/4.0/"
        assert citation["score"] > 0
        assert "superposition" in citation["quote"].lower()

        # 5. Verify citation tool
        verify_res = await session.call_tool(
            "verify_citation",
            {"quote": citation["quote"], "source_url": citation["source_url"]},
        )
        assert not verify_res.is_error
        verify_data = json.loads(verify_res.content[0].text)
        assert verify_data["verified"] is True
        assert verify_data["match_type"] in ("exact", "normalized")
        assert verify_data["match_position"] is not None

        # 6. Unanswerable question -> Abstention
        unans_res = await session.call_tool(
            "answer_with_citations",
            {"question": "What is the secret recipe for traditional French pastry?", "k": 5},
        )
        assert not unans_res.is_error
        unans_data = json.loads(unans_res.content[0].text)
        assert unans_data["answer"] is None
        assert unans_data["abstention_reason"] is not None
        assert "insufficient" in unans_data["abstention_reason"].lower()


@pytest.mark.asyncio
async def test_mcp_error_handling_and_edge_cases(mcp_server_params):
    """Verify error sanitization, empty inputs, non-existent records, and API 500 handling."""
    async with (
        stdio_client(mcp_server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        # Empty query in search
        res = await session.call_tool("search", {"query": "   "})
        data = json.loads(res.content[0].text)
        assert "error" in data
        assert "empty" in data["error"].lower()

        # Empty ID in get_record
        res = await session.call_tool("get_record", {"id": ""})
        data = json.loads(res.content[0].text)
        assert "error" in data

        # Non-existent ID in get_record
        res = await session.call_tool("get_record", {"id": "hal-nonexistent-0000"})
        data = json.loads(res.content[0].text)
        assert "error" in data
        assert "not found" in data["error"].lower()

        # Empty quote in verify_citation
        res = await session.call_tool(
            "verify_citation", {"quote": "", "source_url": "https://hal.science/1"}
        )
        data = json.loads(res.content[0].text)
        assert "error" in data

        # Unknown source URL in verify_citation
        res = await session.call_tool(
            "verify_citation",
            {"quote": "some valid quote", "source_url": "https://unknown.org/doc"},
        )
        data = json.loads(res.content[0].text)
        assert data["verified"] is False
        assert "local store" in data["detail"].lower()

        # HAL API 500 failure handled cleanly without crashing server
        res = await session.call_tool("search", {"query": "error_500"})
        data = json.loads(res.content[0].text)
        assert "error" in data
        assert "HAL API error" in data["error"]


@pytest.mark.asyncio
async def test_mcp_hybrid_retrieval_workflow(mock_hal_server, tmp_path):
    """Verify hybrid multilingual retrieval over stdio MCP protocol.

    Uses DeterministicMockEmbeddingModel in the child server to test cross-lingual
    retrieval and abstention without downloading models or requiring GPUs.
    """
    src_dir = str(Path(__file__).resolve().parent.parent / "src")
    env = {
        "CITECAT_HAL_BASE_URL": mock_hal_server,
        "CITECAT_CACHE_DB_PATH": str(tmp_path / "hybrid_cache.db"),
        "CITECAT_RETRIEVAL_DB_PATH": str(tmp_path / "hybrid_retrieval.db"),
        "CITECAT_RETRIEVAL_MODE": "hybrid",
        "CITECAT_EMBEDDING_MODEL": "mock-multilingual-v1",
        "CITECAT_HYBRID_ALPHA": "0.5",
        "CITECAT_MIN_SIMILARITY_THRESHOLD": "0.30",
        "PYTHONPATH": src_dir,
    }
    for key in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "COMSPEC", "PATHEXT", "PYTHONHOME"):
        val = os.environ.get(key)
        if val is not None:
            env[key] = val

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "citecat.server"],
        env=env,
    )

    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        # 1. Search to ingest the English quantum paper into retrieval store
        search_res = await session.call_tool("search", {"query": "quantum computing", "limit": 2})
        assert not search_res.is_error

        # 2. Ask a French question against the English paper via hybrid retrieval
        french_query = "Comment l'ordinateur quantique effectue-t-il des calculs complexes ?"
        qa_res = await session.call_tool(
            "answer_with_citations", {"question": french_query, "k": 3}
        )
        assert not qa_res.is_error
        qa_data = json.loads(qa_res.content[0].text)
        assert qa_data["answer"] is not None
        assert len(qa_data["citations"]) >= 1
        assert qa_data["citations"][0]["record_id"] == "hal-01234567"
        assert qa_data["citations"][0]["score"] > 0

        # 3. Unanswerable question must strictly abstain under hybrid mode
        unans_res = await session.call_tool(
            "answer_with_citations",
            {"question": "Quelle est la meilleure recette de cuisine française ?", "k": 3},
        )
        assert not unans_res.is_error
        unans_data = json.loads(unans_res.content[0].text)
        assert unans_data["answer"] is None
        assert unans_data["abstention_reason"] is not None


@pytest.mark.asyncio
async def test_mcp_ingest_pdf_tool_over_protocol(mcp_server_params):
    """Exercise ingest_pdf tool over stdio MCP subprocess."""
    async with (
        stdio_client(mcp_server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        # 1. Non-existent record returns clean structured error
        res_err = await session.call_tool("ingest_pdf", {"id": "hal-nonexistent-9999"})
        assert not res_err.is_error
        data_err = json.loads(res_err.content[0].text)
        assert data_err.get("status") == "error"

        # 2. Existing record without PDF returns clean no_pdf_available status
        res_nopdf = await session.call_tool("ingest_pdf", {"id": "hal-01234567"})
        assert not res_nopdf.is_error
        data_nopdf = json.loads(res_nopdf.content[0].text)
        assert data_nopdf.get("status") == "no_pdf_available"
