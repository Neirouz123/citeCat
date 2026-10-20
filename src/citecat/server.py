"""CiteCat MCP server – exposes HAL search and citation tools via FastMCP."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

try:
    from mcp.server.mcpserver import MCPServer

    FastMCP = MCPServer
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore

        MCPServer = FastMCP
    except ImportError as err:
        raise ImportError("Could not import FastMCP or MCPServer from mcp.server") from err

from citecat.cache import Cache
from citecat.citations import (
    async_build_answer_with_citations,
)
from citecat.citations import (
    verify_citation as _verify_citation_quote,
)
from citecat.client import HALClient, HALClientError, make_cache_key
from citecat.config import Config
from citecat.ingest import IngestStore
from citecat.models import BatchIngestResult, Record
from citecat.pdf import (
    PDFContentTypeError,
    PDFDownloader,
    PDFExtractor,
    PDFSecurityError,
    PDFSizeLimitError,
)
from citecat.retrieval import (
    BM25Retriever,
    DeterministicMockEmbeddingModel,
    HybridRetriever,
    Retriever,
    SemanticRetriever,
    SentenceTransformerEmbeddingModel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singletons (initialised lazily on first use)
# ---------------------------------------------------------------------------
_config: Config | None = None
_hal_client: HALClient | None = None
_pdf_downloader: PDFDownloader | None = None
_cache: Cache | None = None
_store: IngestStore | None = None
_retriever: Retriever | None = None
_llm_provider: Any | None = None


def _get_config() -> Config:
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def _get_hal_client() -> HALClient:
    global _hal_client
    if _hal_client is None:
        _hal_client = HALClient(_get_config())
    return _hal_client


def _get_pdf_downloader() -> PDFDownloader:
    global _pdf_downloader
    if _pdf_downloader is None:
        _pdf_downloader = PDFDownloader(_get_config())
    return _pdf_downloader


def _get_cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = Cache(_get_config())
    return _cache


def _get_store() -> IngestStore:
    global _store
    if _store is None:
        _store = IngestStore(_get_config())
    return _store


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        cfg = _get_config()
        mode = cfg.retrieval_mode.lower()
        if mode in ("hybrid", "semantic"):
            try:
                if cfg.embedding_model.startswith("mock-"):
                    model = DeterministicMockEmbeddingModel(model_name=cfg.embedding_model)
                else:
                    model = SentenceTransformerEmbeddingModel(
                        model_name=cfg.embedding_model,
                        cache_folder=cfg.embedding_cache_dir or None,
                    )
                store = _get_store()
                sem_retriever = SemanticRetriever(model, store=store)
                if mode == "semantic":
                    _retriever = sem_retriever
                else:
                    _retriever = HybridRetriever(
                        lexical_retriever=BM25Retriever(),
                        semantic_retriever=sem_retriever,
                        alpha=cfg.hybrid_alpha,
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to initialize %s retriever (%s). Falling back to BM25.",
                    mode,
                    exc,
                )
                _retriever = BM25Retriever()
        else:
            _retriever = BM25Retriever()
    return _retriever


def _get_llm_provider() -> Any | None:
    global _llm_provider
    if _llm_provider is None:
        cfg = _get_config()
        if cfg.synthesis_mode.lower() == "llm":
            if cfg.llm_provider.lower() in ("openai", "custom"):
                from citecat.llm import OpenAICompatibleProvider

                _llm_provider = OpenAICompatibleProvider(cfg)
            elif cfg.llm_provider.lower() == "fake":
                from citecat.llm import FakeLLMProvider

                _llm_provider = FakeLLMProvider()
    return _llm_provider


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------
mcp_server = MCPServer("citecat", version="0.1.0")


@mcp_server.tool()
async def search(query: str, filters: dict[str, str] | None = None, limit: int = 10) -> str:
    """Search HAL open-science publications.

    **Inputs:**
    - query (str): Free-text search query for scientific publications.
    - filters (dict, optional): Solr filter queries, e.g. {"docType_s": "ART"}.
    - limit (int): Maximum results to return (1–100, default 10).

    **Output:** JSON array of records with id, title, snippet, source_url,
    license, authors, language, and publication_date.

    **Limitations:** Results come from the HAL API. Availability depends on
    network access. Snippets are abstracts when available; some records lack
    abstracts. License reflects the HAL metadata; 'unknown' means HAL did
    not specify a license.
    """
    if not query or not query.strip():
        return json.dumps({"error": "Query must not be empty."})
    limit = max(1, min(limit, 100))

    cache = _get_cache()
    cache_key = make_cache_key(query, limit, filters)

    # Check cache
    cached = cache.get(cache_key)
    if cached is not None:
        return json.dumps(cached)

    try:
        client = _get_hal_client()
        records = await client.search(query, limit=limit, filters=filters)
    except HALClientError as exc:
        return json.dumps({"error": f"HAL API error: {exc}"})
    except Exception as exc:
        logger.exception("Unexpected error in search")
        return json.dumps({"error": f"Unexpected error: {type(exc).__name__}"})

    data = [r.model_dump(mode="json") for r in records]
    cache.put(cache_key, data)

    # Also ingest into retrieval store for RAG
    store = _get_store()
    for record in records:
        store.ingest_record(record)

    return json.dumps(data)


@mcp_server.tool()
async def get_record(id: str) -> str:
    """Fetch a single HAL record by its identifier.

    **Inputs:**
    - id (str): HAL document identifier, e.g. 'hal-01234567'.

    **Output:** JSON object with record fields, or an error if not found.

    **Limitations:** Requires network access to HAL. Returns null fields for
    metadata HAL does not provide.
    """
    if not id or not id.strip():
        return json.dumps({"error": "Record ID must not be empty."})

    try:
        client = _get_hal_client()
        record = await client.fetch(id)
    except HALClientError as exc:
        return json.dumps({"error": f"HAL API error: {exc}"})
    except Exception as exc:
        logger.exception("Unexpected error in get_record")
        return json.dumps({"error": f"Unexpected error: {type(exc).__name__}"})

    if record is None:
        return json.dumps({"error": f"Record '{id}' not found on HAL."})

    # Ingest for RAG
    store = _get_store()
    store.ingest_record(record)

    return json.dumps(record.model_dump(mode="json"))


@mcp_server.tool()
async def answer_with_citations(question: str, k: int = 5) -> str:
    """Answer a question using ingested HAL publications with verifiable citations.

    **Inputs:**
    - question (str): The research question to answer.
    - k (int): Number of top passages to consider (default 5).

    **Output:** JSON with 'answer' (string or null), 'citations' (array of
    objects with quote, source_url, license, record_id, score), and
    'abstention_reason' (string or null).

    **Retrieval Strategy:** Configured via `CITECAT_RETRIEVAL_MODE` ('bm25',
    'hybrid', or 'semantic'). Hybrid mode combines Okapi BM25 with multilingual
    dense embeddings (e.g., BAAI/bge-m3), enabling French-English cross-lingual
    retrieval and semantic paraphrasing. Scores are calibrated and normalized.

    **Limitations:** Only searches documents previously ingested via search or
    get_record. Returns verbatim passages as evidence with exact quotes rather
    than ungrounded generative text. Abstains when evidence is insufficient
    rather than fabricating answers or citations.
    """
    if not question or not question.strip():
        return json.dumps({"error": "Question must not be empty."})
    k = max(1, min(k, 20))

    cfg = _get_config()
    store = _get_store()
    retriever = _get_retriever()
    provider = _get_llm_provider()
    min_score = (
        cfg.min_similarity_threshold
        if cfg.retrieval_mode.lower() in ("hybrid", "semantic")
        else 0.5
    )
    result = await async_build_answer_with_citations(
        question,
        store,
        retriever,
        k=k,
        min_score=min_score,
        provider=provider,
        synthesis_mode=cfg.synthesis_mode,
        config=cfg,
    )
    return json.dumps(result.model_dump(mode="json"))


@mcp_server.tool(name="verify_citation")
async def verify_citation(quote: str, source_url: str) -> str:
    """Verify whether a quote actually appears in a source document.

    **Inputs:**
    - quote (str): The exact text claimed to be from the source.
    - source_url (str): The canonical URL (or ID) of the source document.

    **Output:** JSON with 'verified' (bool), 'match_type' ('exact' or
    'normalized'), 'match_position' (int or null), and 'detail' (string).

    **Limitations:** Only checks against documents already ingested into the
    local store. Uses exact matching first, then conservative normalized
    matching (Unicode NFKC, diacritics, typography quotes, case folding,
    whitespace collapsing). Does not fetch external URLs over HTTP; it checks
    the locally stored source text. A successful verification means the quote
    exists in the source text, not necessarily that it supports a particular
    factual claim.
    """
    if not quote or not quote.strip():
        return json.dumps({"error": "Quote must not be empty."})
    if not source_url or not source_url.strip():
        return json.dumps({"error": "Source URL must not be empty."})

    store = _get_store()
    conn = store._get_conn()
    row = conn.execute(
        "SELECT full_text FROM records WHERE source_url = ? OR id = ?",
        (source_url, source_url),
    ).fetchone()

    if row is None:
        return json.dumps(
            {
                "verified": False,
                "match_type": None,
                "match_position": None,
                "detail": (
                    f"No document with source_url or ID '{source_url}' found in local store. "
                    "Try searching or fetching the record first."
                ),
            }
        )

    source_text = row[0]
    result = _verify_citation_quote(quote, source_text)
    return json.dumps(result.model_dump(mode="json"))


@mcp_server.resource("catalogue://record/{id}")
async def catalogue_record(id: str) -> str:
    """Access a HAL record as an MCP resource.

    Returns the record's JSON representation from the local store, or
    fetches it from HAL if not yet ingested.
    """
    store = _get_store()
    meta = store.get_record_metadata(id)
    if meta is not None:
        full_text = store.get_record_text(id) or ""
        meta["full_text_preview"] = full_text[:500] if full_text else ""
        return json.dumps(meta)

    # Try fetching from HAL
    try:
        client = _get_hal_client()
        record = await client.fetch(id)
    except Exception:
        return json.dumps({"error": f"Could not retrieve record '{id}'."})

    if record is None:
        return json.dumps({"error": f"Record '{id}' not found."})

    store.ingest_record(record)
    return json.dumps(record.model_dump(mode="json"))


@mcp_server.tool(name="ingest_pdf")
async def ingest_pdf(id: str) -> str:
    """Retrieve and ingest the full text of an open-access PDF for a HAL record.

    **Inputs:**
    - id (str): HAL document identifier (e.g. 'hal-01234567').

    **Output:** JSON object with status ('success', 'image_only', 'encrypted',
    'empty', 'corrupt', 'download_failed', 'no_pdf_available'), pages_extracted,
    passages_stored, ocr_required, pdf_url, and error details if any.

    **Behavior:**
    Fetches the HAL record, checks for open-access PDF availability, downloads the
    file safely (enforcing size limits, timeouts, and Content-Type checks),
    extracts text page-by-page preserving page numbers, and stores passages in the
    retrieval engine for BM25 and semantic search.

    **Limitations:**
    Only openly accessible documents can be retrieved. If the document is scanned
    or image-only, OCR is detected as required rather than inventing text.
    """
    if not id or not id.strip():
        return json.dumps({"error": "Record ID must not be empty."})

    clean_id = id.strip()
    client = _get_hal_client()
    store = _get_store()
    downloader = _get_pdf_downloader()

    try:
        record = await client.fetch(clean_id)
    except Exception as exc:
        return json.dumps(
            {
                "status": "error",
                "record_id": clean_id,
                "error": f"Failed to fetch record from HAL: {exc}",
            }
        )

    if record is None:
        return json.dumps(
            {
                "status": "error",
                "record_id": clean_id,
                "error": f"Record '{clean_id}' not found on HAL.",
            }
        )

    store.ingest_record(record)

    if not record.pdf_url:
        return json.dumps(
            {
                "status": "no_pdf_available",
                "record_id": clean_id,
                "error_message": (
                    f"Record '{clean_id}' does not have an open-access PDF deposited on HAL."
                ),
                "pages_extracted": 0,
                "passages_stored": 0,
                "ocr_required": False,
            }
        )

    try:
        pdf_bytes = await downloader.download(record.pdf_url)
    except PDFSecurityError as exc:
        return json.dumps(
            {
                "status": "security_violation",
                "record_id": clean_id,
                "pdf_url": record.pdf_url,
                "error_message": f"Security violation: {exc}",
                "pages_extracted": 0,
                "passages_stored": 0,
                "ocr_required": False,
            }
        )
    except PDFSizeLimitError as exc:
        return json.dumps(
            {
                "status": "download_failed",
                "record_id": clean_id,
                "pdf_url": record.pdf_url,
                "error_message": f"Size limit exceeded: {exc}",
                "pages_extracted": 0,
                "passages_stored": 0,
                "ocr_required": False,
            }
        )
    except PDFContentTypeError as exc:
        return json.dumps(
            {
                "status": "download_failed",
                "record_id": clean_id,
                "pdf_url": record.pdf_url,
                "error_message": f"Invalid content type: {exc}",
                "pages_extracted": 0,
                "passages_stored": 0,
                "ocr_required": False,
            }
        )
    except Exception as exc:
        return json.dumps(
            {
                "status": "download_failed",
                "record_id": clean_id,
                "pdf_url": record.pdf_url,
                "error_message": f"Download failed: {exc}",
                "pages_extracted": 0,
                "passages_stored": 0,
                "ocr_required": False,
            }
        )

    cfg = _get_config()
    extraction = PDFExtractor.extract_text(
        pdf_bytes,
        record_id=clean_id,
        pdf_url=record.pdf_url,
        max_pages=cfg.pdf_max_pages,
        max_size_bytes=cfg.pdf_max_size_bytes,
    )

    if extraction.status == "success":
        passages_count = store.ingest_pdf(record, extraction)
        return json.dumps(
            {
                "status": "success",
                "record_id": clean_id,
                "pdf_url": record.pdf_url,
                "total_pages": extraction.total_pages,
                "pages_extracted": extraction.extracted_pages,
                "total_words": extraction.total_words,
                "passages_stored": passages_count,
                "ocr_required": False,
            }
        )
    else:
        return json.dumps(
            {
                "status": extraction.status,
                "record_id": clean_id,
                "pdf_url": record.pdf_url,
                "total_pages": extraction.total_pages,
                "pages_extracted": extraction.extracted_pages,
                "total_words": extraction.total_words,
                "passages_stored": 0,
                "ocr_required": extraction.ocr_required,
                "error_message": extraction.error_message,
            }
        )


async def ingest_batch_records(
    query: str | None = None,
    record_ids: list[str] | None = None,
    limit: int = 5,
    config: Config | None = None,
) -> BatchIngestResult:
    """Bounded batch ingestion of HAL records with full-text PDFs."""
    cfg = config or _get_config()
    client = HALClient(cfg)
    store = IngestStore(cfg)
    downloader = PDFDownloader(cfg)

    result = BatchIngestResult()
    records_to_process: list[Record] = []

    try:
        if record_ids:
            bounded_ids = record_ids[:limit]
            for rid in bounded_ids:
                rec = await client.fetch(rid)
                if rec:
                    records_to_process.append(rec)
                else:
                    result.records_processed += 1
                    result.pdfs_skipped += 1
                    result.skip_reasons["not_found"] = result.skip_reasons.get("not_found", 0) + 1
                    result.details.append({"id": rid, "status": "record_not_found"})
        elif query:
            bounded_limit = max(1, min(limit, 50))
            records_to_process = await client.search(query, limit=bounded_limit)

        for rec in records_to_process:
            result.records_processed += 1
            if not rec.pdf_url:
                result.pdfs_skipped += 1
                result.skip_reasons["no_pdf_url"] = result.skip_reasons.get("no_pdf_url", 0) + 1
                result.details.append({"id": rec.id, "status": "no_pdf_available"})
                store.ingest_record(rec)
                continue

            try:
                pdf_bytes = await downloader.download(rec.pdf_url)
                result.pdfs_downloaded += 1
            except Exception as exc:
                result.extraction_failures += 1
                result.details.append(
                    {
                        "id": rec.id,
                        "status": "download_failed",
                        "error": str(exc),
                    }
                )
                store.ingest_record(rec)
                continue

            extraction = PDFExtractor.extract_text(pdf_bytes, record_id=rec.id, pdf_url=rec.pdf_url)
            if extraction.status == "success":
                p_count = store.ingest_pdf(rec, extraction)
                result.pages_extracted += extraction.extracted_pages
                result.passages_stored += p_count
                result.details.append(
                    {
                        "id": rec.id,
                        "status": "success",
                        "pages": extraction.extracted_pages,
                        "passages": p_count,
                    }
                )
            elif extraction.status == "image_only":
                result.pdfs_skipped += 1
                result.skip_reasons["image_only_ocr_required"] = (
                    result.skip_reasons.get("image_only_ocr_required", 0) + 1
                )
                result.details.append(
                    {
                        "id": rec.id,
                        "status": "image_only",
                        "ocr_required": True,
                    }
                )
                store.ingest_record(rec)
            else:
                result.extraction_failures += 1
                result.details.append(
                    {
                        "id": rec.id,
                        "status": extraction.status,
                        "error": extraction.error_message,
                    }
                )
                store.ingest_record(rec)

    finally:
        await client.close()
        await downloader.close()
        store.close()

    return result


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
async def _cleanup() -> None:
    """Clean up async resources."""
    global _hal_client, _pdf_downloader, _llm_provider, _cache, _store
    if _hal_client is not None:
        await _hal_client.close()
        _hal_client = None
    if _pdf_downloader is not None:
        await _pdf_downloader.close()
        _pdf_downloader = None
    if _llm_provider is not None:
        await _llm_provider.close()
        _llm_provider = None
    if _cache is not None:
        _cache.close()
        _cache = None
    if _store is not None:
        _store.close()
        _store = None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the CiteCat MCP server or CLI ingestion commands."""
    import argparse
    import atexit
    import sys

    parser = argparse.ArgumentParser(
        prog="citecat",
        description="CiteCat: MCP server and full-text PDF research client for HAL",
    )
    subparsers = parser.add_subparsers(dest="command")

    # citecat ingest-pdf <record_id>
    pdf_cmd = subparsers.add_parser(
        "ingest-pdf", help="Download and ingest full-text PDF for a HAL record"
    )
    pdf_cmd.add_argument("record_id", help="HAL document identifier")

    # citecat ingest-batch --query "..." --limit 5
    batch_cmd = subparsers.add_parser(
        "ingest-batch", help="Bounded batch ingestion of records and PDFs"
    )
    batch_cmd.add_argument("--query", "-q", help="HAL search query")
    batch_cmd.add_argument("--ids", help="Comma-separated HAL IDs (e.g. hal-01,hal-02)")
    batch_cmd.add_argument(
        "--limit",
        "-l",
        type=int,
        default=5,
        help="Maximum records to process (default 5)",
    )

    args = parser.parse_args()

    if args.command == "ingest-pdf":
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )
        print(f"Ingesting PDF for HAL record: {args.record_id}...")
        res = asyncio.run(ingest_pdf(args.record_id))
        print(json.dumps(json.loads(res), indent=2))
        return

    if args.command == "ingest-batch":
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )
        ids_list = [i.strip() for i in args.ids.split(",")] if args.ids else None
        print(
            f"Starting bounded batch ingestion (limit={args.limit}, "
            f"query={args.query!r}, ids={ids_list})..."
        )
        res_obj = asyncio.run(
            ingest_batch_records(query=args.query, record_ids=ids_list, limit=args.limit)
        )
        print("\n--- Ingestion Report ---")
        print(json.dumps(res_obj.model_dump(mode="json"), indent=2))
        return

    # Default: run stdio MCP server
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    logger.info("Starting CiteCat MCP server...")
    sys.stderr.flush()

    # Register cleanup
    def cleanup_sync() -> None:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_cleanup())
            else:
                loop.run_until_complete(_cleanup())
        except RuntimeError:
            pass

    atexit.register(cleanup_sync)

    mcp_server.run(transport="stdio")


if __name__ == "__main__":
    main()
