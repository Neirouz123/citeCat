"""Bounded live smoke test against official HAL API endpoint.

This test requires active internet access and is marked with `@pytest.mark.live`.
It is excluded from ordinary offline test runs by default.
Run explicitly with: `uv run pytest -m live`
"""

from __future__ import annotations

import pytest

from citecat.citations import build_answer_with_citations, verify_citation
from citecat.client import HALClient
from citecat.config import Config
from citecat.ingest import IngestStore


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_hal_search_fetch_and_citation_pipeline(tmp_path):
    """Smoke test: search live HAL, fetch a real record, and verify citations."""
    env_cfg = Config.from_env()
    config = Config(
        hal_timeout=15.0,
        rate_limit_per_second=2.0,
        hal_ssl_verify=env_cfg.hal_ssl_verify,
        retrieval_db_path=str(tmp_path / "live_retrieval.db"),
    )
    client = HALClient(config)
    store = IngestStore(config)

    try:
        # 1. Search HAL for scientific papers on "quantum"
        records = await client.search("quantum", limit=2)
        assert len(records) >= 1, "Expected at least 1 record from live HAL search"

        rec = records[0]
        # Validate provenance & metadata
        assert rec.id, "Record must have an id"
        assert rec.title, "Record must have a title"
        assert rec.source_url.startswith("http"), "Source URL must be a valid HTTP URL"
        assert rec.license, "License must be present (or 'unknown')"
        assert rec.retrieved_at is not None

        # 2. Fetch the specific record by HAL ID
        fetched = await client.fetch(rec.id)
        assert fetched is not None, f"Failed to fetch record {rec.id} from HAL"
        assert fetched.id == rec.id
        assert fetched.source_url == rec.source_url

        # 3. Test citation & verification pipeline on fetched document
        if fetched.snippet and len(fetched.snippet.split()) >= 10:
            store.ingest_record(fetched)
            # Pick a word from the title to query
            query = fetched.title.split()[0]
            ans = build_answer_with_citations(query, store, min_score=0.01)

            if ans.citations:
                quote = ans.citations[0].quote
                v_res = verify_citation(quote, fetched.snippet)
                assert v_res.verified, "Extracted citation quote must verify against source text"

        # 4. Verify fetching a non-existent identifier returns None
        non_existent = await client.fetch("hal-nonexistent-9999999999")
        assert non_existent is None

    finally:
        await client.close()
        store.close()


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_mcp_protocol_workflow(tmp_path):
    """Smoke test: exercise CiteCat MCP server subprocess over stdio against live HAL API."""
    import json
    import os
    import sys
    from pathlib import Path

    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    src_dir = str(Path(__file__).resolve().parent.parent / "src")
    env = {
        "CITECAT_HAL_BASE_URL": "https://api.archives-ouvertes.fr",
        "CITECAT_CACHE_DB_PATH": str(tmp_path / "live_protocol_cache.db"),
        "CITECAT_RETRIEVAL_DB_PATH": str(tmp_path / "live_protocol_retrieval.db"),
        "CITECAT_RATE_LIMIT_PER_SECOND": "2.0",
        "PYTHONPATH": src_dir,
    }
    for k, v in os.environ.items():
        if k.startswith("CITECAT_"):
            env[k] = v
    for key in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "COMSPEC", "PATHEXT", "PYTHONHOME"):
        val = os.environ.get(key)
        if val is not None:
            env[key] = val

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "citecat.server"],
        env=env,
    )

    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        init_res = await session.initialize()
        assert init_res.server_info.name == "citecat"

        # 1. Search live HAL via MCP protocol
        search_res = await session.call_tool("search", {"query": "deep learning", "limit": 2})
        assert not search_res.is_error
        records = json.loads(search_res.content[0].text)
        assert isinstance(records, list)
        assert len(records) >= 1

        target_id = records[0]["id"]
        assert target_id

        # 2. Get record via MCP protocol
        get_res = await session.call_tool("get_record", {"id": target_id})
        assert not get_res.is_error
        record = json.loads(get_res.content[0].text)
        assert record["id"] == target_id

        # 3. Read record resource via MCP protocol
        res_read = await session.read_resource(f"catalogue://record/{target_id}")
        assert len(res_read.contents) >= 1
        res_json = json.loads(res_read.contents[0].text)
        assert res_json["id"] == target_id


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_hal_pdf_ingestion_and_citation(tmp_path):
    """Smoke test: search open-access PDF, download, extract, ingest, and verify citation."""
    from citecat.pdf import PDFDownloader, PDFExtractor

    env_cfg = Config.from_env()
    config = Config(
        hal_timeout=30.0,
        pdf_timeout=30.0,
        rate_limit_per_second=2.0,
        hal_ssl_verify=env_cfg.hal_ssl_verify,
        retrieval_db_path=str(tmp_path / "live_pdf_retrieval.db"),
        pdf_cache_dir=str(tmp_path / "live_pdf_cache"),
    )
    client = HALClient(config)
    downloader = PDFDownloader(config)
    store = IngestStore(config)

    try:
        # 1. Search for open-access documents with deposited files
        records = await client.search(
            "openAccess_bool:true AND submitType_s:file",
            limit=5,
            filters={"docType_s": "ART"},
        )
        assert len(records) >= 1

        # Find first record with a valid PDF URL
        target_rec = next((r for r in records if r.pdf_url), None)
        if target_rec is None:
            pytest.skip("No open-access record with direct PDF URL returned in top 5")

        # 2. Download PDF safely
        pdf_bytes = await downloader.download(target_rec.pdf_url)
        assert len(pdf_bytes) > 0
        assert b"%PDF-" in pdf_bytes[:1024]

        # 3. Extract text page-by-page
        extraction = PDFExtractor.extract_text(
            pdf_bytes, record_id=target_rec.id, pdf_url=target_rec.pdf_url
        )
        assert extraction.total_pages >= 1

        if extraction.status == "success":
            # 4. Ingest PDF passages
            count = store.ingest_pdf(target_rec, extraction)
            assert count >= 1

            passages = store.get_all_passages()
            assert len(passages) >= 1
            assert passages[0].source_type == "pdf"
            assert passages[0].page_number >= 1

            # 5. Build answer from first passage words and verify
            query_words = " ".join(passages[0].text.split()[:4])
            ans = build_answer_with_citations(query_words, store, min_score=0.01)
            if ans.citations:
                v_res = verify_citation(
                    ans.citations[0].quote, store.get_record_text(target_rec.id)
                )
                assert v_res.verified is True

    finally:
        await client.close()
        await downloader.close()
        store.close()
