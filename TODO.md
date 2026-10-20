# CiteCat – TODO

## Completed in Audit

- [x] Concurrency-safe rate limiter with `asyncio.Lock` and scheduled interval distribution
- [x] Thread-safe SQLite access for `Cache` and `IngestStore` with `threading.Lock()`
- [x] Fixed passage splitting character offset calculation using regex word token spans (`text[start:end] == passage_text`)
- [x] Added stop-word filtering to `BM25Retriever` to prevent stop-word false positive matches
- [x] Implemented conservative normalized citation verification handling diacritics and typography variants
- [x] Added comprehensive MCP tool and resource execution tests via `mcp_server.call_tool` and `mcp_server.read_resource`
- [x] Added live bounded HAL smoke test (`tests/test_live_smoke.py`, gated with `@pytest.mark.live`)
- [x] Verified packaging with `uv build` (sdist, wheel, `citecat = citecat.server:main` entry point)
- [x] Implemented deterministic evaluation tests in `tests/test_eval.py` measuring actual results
- [x] **Phase 2 — Multilingual Semantic Search**: Calibrated hybrid retrieval combining Okapi BM25 and multilingual dense embeddings (`BAAI/bge-m3`), IEEE 754 float32 vector BLOB persistence, staleness hash validation, and bilingual benchmark harness
- [x] **Phase 3 — Full-Text PDF Ingestion from HAL**: Safe bounded streaming PDF download, size and redirect limits, `%PDF-` validation, disk caching, PyMuPDF page-by-page text extraction, 1-indexed page number provenance (`[Title — Page N]`), scanned/OCR detection (`ocr_required=True`), bounded CLI subcommands (`ingest-pdf`, `ingest-batch`), and full-text evaluation suite
- [x] **Phase 3 Hardening & Security Audit**: Strict SSRF protections blocking localhost, private, and link-local cloud metadata IPs; per-hop redirect validation; atomic temporary-file cache writes; corrupted cache self-healing; decompression bomb protection with `pdf_max_pages`; defensive non-success extraction guards in ingestion store
- [x] **Phase 4 — Grounded LLM Answer Synthesis**: Optional provider abstraction (`LLMProvider`, `OpenAICompatibleProvider`, `FakeLLMProvider`), strict evidence bounding with `<evidence>` tags and prompt injection defenses, claim-level quote verification against stored text, stored provenance enforcement for URLs/licenses/pages, safe abstention on unsupported questions/timeouts/errors, and backwards-compatible MCP server integration
- [x] **Phase 5 — Real-World Validation & Release Readiness**: Opt-in live LLM integration test harness (`tests/test_live_llm.py` with `@pytest.mark.live_llm`), live provider auth/timeout/error/secret-redaction verification, 18-question Phase 5 quality benchmark (`tests/eval_phase5.py`), extended French typography and ligature normalization (`œ`, `æ`, narrow no-break space `\u202f`), end-to-end claim-to-citation provenance audit, 181 passing offline tests, clean Ruff checks, and release package build
- [ ] **Phase 6 — Real LLM End-to-End Acceptance**: Acceptance test harness with 7 real-model scenarios implemented in `tests/test_live_llm.py`. Network failure and secret redaction tests pass (4 passed); live model inference tests are blocked pending user-provided credentials or local Ollama daemon.

## Remaining Work (Prioritized)

### High priority
1. **Fuzzy Citation Verification with Levenshtein / Token Distance**: Support small OCR or transcription errors with a configurable similarity threshold (e.g. 95% similarity).

### Medium priority
3. **Passage Deduplication**: Deduplicate passages across multiple revisions or preprint/journal versions of the same publication.
4. **Cursor-based Search Pagination in MCP Tool**: Expose `start` offset or pagination cursor through the `search` MCP tool.
5. **Structured JSON Logging & Metrics**: Add structured logging and telemetry for production deployment monitoring.
6. **Async SQLite Driver**: Switch to `aiosqlite` for non-blocking database queries under high concurrency.

### Low priority / Nice to have
7. **Phase 6 A2A Agent Integration**: Provide a standalone A2A agent card and transport adapter isolated from the core MCP package.
8. **Export Formats**: Add BibTeX, RIS, and CSL-JSON citation export utilities.
