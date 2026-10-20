# CiteCat 🐱📚

**Production-minded MCP server for searching HAL scientific publications with verifiable citations, multilingual semantic retrieval, and full-text PDF ingestion.**

CiteCat is a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that provides AI assistants with access to [HAL](https://hal.science/) — France's open-science archive — ensuring that every factual claim is traceable to source text and every quotation is independently verifiable across French and English scientific publications down to exact page numbers and verbatim passages.

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                      MCP Client                        │
│             (Claude Desktop, Cursor, etc.)             │
└───────────────────────────┬────────────────────────────┘
                            │ stdio transport (JSON-RPC)
                            ▼
┌────────────────────────────────────────────────────────┐
│                   CiteCat MCPServer                    │
│             (search, get_record, ingest_pdf,           │
│           answer_with_citations, verify_citation)      │
└──────────────┬──────────────────────────┬──────────────┘
               │                          │
               ▼                          ▼
┌─────────────────────────┐  ┌───────────────────────────┐
│  SQLite Response Cache  │  │   Retrieval Engine        │
│      (SHA-256 + TTL)    │  │ ┌───────────────────────┐ │
└──────────────┬──────────┘  │ │ Okapi BM25 Lexical    │ │
               │             │ └──────────┬────────────┘ │
               ▼             │            ▼              │
┌─────────────────────────┐  │ ┌───────────────────────┐ │
│      HAL Solr API       │  │ │ Calibrated Hybrid     │ │
│  (api.archives-ouvertes)│  │ │ Ranker (alpha weight) │ │
└──────────────┬──────────┘  │ └──────────▲────────────┘ │
               │             │            │              │
               ▼             │ ┌──────────┴────────────┐ │
┌─────────────────────────┐  │ │ Multilingual Dense    │ │
│  PDF Downloader (Safe)  │  │ │ Embeddings (BGE-M3)   │ │
│ (size limits, %PDF-,    │  │ └───────────────────────┘ │
│  disk cache, SHA-256)   │  └─────────────┬─────────────┘
└──────────────┬──────────┘                │
               │                           │
               ▼                           │
┌─────────────────────────┐                │
│ PyMuPDF Text Extractor  │                │
│ (page tracking, offsets,│                │
│  scanned OCR detection) │                │
└──────────────┬──────────┘                │
               │                           │
               ▼                           ▼
┌────────────────────────────────────────────────────────┐
│                      SQLite Store                      │
│      (Records, Passages with Page Numbers & Offsets,   │
│           Full Text, Binary Float32 Vector Blobs)      │
└────────────────────────────────────────────────────────┘
```

### Module Responsibilities

| Module | File | Purpose |
|---|---|---|
| Configuration | [`src/citecat/config.py`](file:///c:/Users/windows/Desktop/citeCat/src/citecat/config.py) | Environment configuration: retrieval mode, model IDs, alpha weights, thresholds, timeouts, PDF download limits, SSL verify |
| Data Models | [`src/citecat/models.py`](file:///c:/Users/windows/Desktop/citeCat/src/citecat/models.py) | Pydantic v2 models: `Record`, `Passage`, `Citation`, `AnswerResult`, `VerificationResult`, `ExtractedPage`, `PDFExtractionResult`, `BatchIngestResult` |
| HAL Client | [`src/citecat/client.py`](file:///c:/Users/windows/Desktop/citeCat/src/citecat/client.py) | Async client with rate-limiting mutex, concurrency bounds, exponential backoff, open-access PDF URL resolution (`fileMain_s`, `files_s`) |
| PDF Downloader & Extractor | [`src/citecat/pdf.py`](file:///c:/Users/windows/Desktop/citeCat/src/citecat/pdf.py) | Safe streaming PDF downloads, size limits, magic bytes validation, disk caching, PyMuPDF page-by-page extraction, and scanned OCR detection |
| Cache | [`src/citecat/cache.py`](file:///c:/Users/windows/Desktop/citeCat/src/citecat/cache.py) | Thread-safe SQLite response cache with SHA-256 keys and configurable TTL |
| Ingestion & Storage | [`src/citecat/ingest.py`](file:///c:/Users/windows/Desktop/citeCat/src/citecat/ingest.py) | Abstract & full-text PDF passage splitting (`full_text[start:end] == passage`), page tracking, vector BLOB storage, and staleness validation |
| Retrieval | [`src/citecat/retrieval.py`](file:///c:/Users/windows/Desktop/citeCat/src/citecat/retrieval.py) | BM25 lexical with French elision support, dense semantic (`SentenceTransformerEmbeddingModel`, `DeterministicMockEmbeddingModel`), and hybrid retrieval |
| Citations | [`src/citecat/citations.py`](file:///c:/Users/windows/Desktop/citeCat/src/citecat/citations.py) | Structured answer generation, page-level citation attribution (`[Title — Page N]`), calibrated abstention, and verbatim quote verification |
| MCP Server & CLI | [`src/citecat/server.py`](file:///c:/Users/windows/Desktop/citeCat/src/citecat/server.py) | MCP server registration over stdio, bounded CLI subcommands (`ingest-pdf`, `ingest-batch`, `serve`), and `catalogue://record/{id}` resource |

---

## Installation & Setup

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

### 1. Standard (BM25 Lexical Baseline)

The default installation is lightweight and has zero machine-learning dependencies:

```bash
# Clone the repository
git clone <repository-url>
cd citeCat

# Install base dev dependencies
uv sync --extra dev

# Run all 121 offline unit & protocol tests
uv run pytest -v

# Run live HAL smoke tests (requires network access)
uv run pytest -m live -v
```

### 2. Optional Multilingual Semantic Search (`[semantic]`)

To enable dense neural retrieval with multilingual models like `BAAI/bge-m3`:

```bash
uv sync --extra semantic --extra dev
# Or using pip
pip install "citecat[semantic,dev]"
```

Hardware requirements for neural models:
- **`BAAI/bge-m3`** (Default): ~2.2 GB RAM/VRAM on CPU/CUDA, 1024 embedding dimensions.
- **`paraphrase-multilingual-MiniLM-L12-v2`** (Lightweight): ~450 MB RAM, 384 dimensions.
- **`mock-multilingual-v1`** (Testing & Offline): Built-in deterministic model, 0 MB download, sub-millisecond execution.

### 3. Optional Full-Text PDF Ingestion (`[pdf]`)

To enable open-access PDF downloading and PyMuPDF text extraction:

```bash
uv sync --extra pdf --extra dev
# Or using pip
pip install "citecat[pdf,dev]"
```

### 4. Full Installation (`[all]`)

To install CiteCat with both multilingual semantic search and PDF ingestion:

```bash
uv sync --extra all
# Or using pip
pip install "citecat[all]"
```

---

## Configuration

All settings are configured via environment variables prefixed with `CITECAT_`:

| Variable | Default | Description |
|---|---|---|
| `CITECAT_RETRIEVAL_MODE` | `bm25` | Retrieval strategy: `bm25`, `hybrid`, or `semantic` |
| `CITECAT_EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model identifier (`sentence-transformers` compatible or `mock-multilingual-v1`) |
| `CITECAT_HYBRID_ALPHA` | `0.5` | Weight for lexical vs semantic scoring ($0.0$ = pure semantic, $1.0$ = pure BM25) |
| `CITECAT_MIN_SIMILARITY_THRESHOLD` | `0.35` | Minimum calibrated score threshold for hybrid/semantic answer generation |
| `CITECAT_EMBEDDING_CACHE_DIR` | `""` | Custom cache directory for downloading and loading model weights |
| `CITECAT_AUTO_EMBED_ON_INGEST` | `true` | Automatically calculate embeddings upon passage ingestion |
| `CITECAT_HAL_BASE_URL` | `https://api.archives-ouvertes.fr` | Official HAL Solr API base URL |
| `CITECAT_HAL_TIMEOUT` | `30.0` | HTTP request timeout in seconds |
| `CITECAT_HAL_MAX_CONCURRENT` | `5` | Maximum concurrent requests to HAL |
| `CITECAT_HAL_MAX_RESULTS` | `100` | Maximum allowed search results cap |
| `CITECAT_RATE_LIMIT_PER_SECOND` | `5.0` | Maximum requests per second to HAL API |
| `CITECAT_MAX_RETRIES` | `3` | Maximum retries for transient HTTP errors (429, 5xx) |
| `CITECAT_RETRY_BASE_DELAY` | `1.0` | Initial exponential backoff delay in seconds |
| `CITECAT_RETRY_MAX_DELAY` | `60.0` | Maximum exponential backoff delay in seconds |
| `CITECAT_CACHE_DB_PATH` | `citecat_cache.db` | SQLite cache database file path |
| `CITECAT_CACHE_TTL_SECONDS` | `3600` | Cache time-to-live in seconds (1 hour) |
| `CITECAT_RETRIEVAL_DB_PATH` | `citecat_retrieval.db` | SQLite retrieval store database file path |
| `CITECAT_HAL_SSL_VERIFY` | `true` | Verify SSL certificates for HAL and PDF downloads (`false` for corporate inspection proxies) |
| `CITECAT_PDF_TIMEOUT` | `30.0` | Timeout in seconds for PDF downloads |
| `CITECAT_PDF_MAX_SIZE_BYTES` | `52428800` | Maximum PDF download size (default 50 MB) |
| `CITECAT_PDF_CACHE_DIR` | `""` | Directory for raw downloaded PDF file cache (default `.citecat_cache/pdfs`) |
| `CITECAT_PDF_MAX_REDIRECTS` | `5` | Maximum HTTP redirect hops allowed for PDF retrieval |
| `CITECAT_PDF_AUTO_DOWNLOAD` | `false` | Automatically download and ingest PDFs during search (default `false` for bounded control) |
| `CITECAT_PDF_ALLOW_PRIVATE_IPS` | `false` | Allow downloads from private/local IP addresses (strictly `false` in production for SSRF prevention) |
| `CITECAT_PDF_MAX_PAGES` | `1000` | Maximum pages processed per document to guard against decompression bombs |
| `CITECAT_SYNTHESIS_MODE` | `extractive` | Synthesis mode for `answer_with_citations`: `extractive` (default, deterministic) or `llm` |
| `CITECAT_LLM_PROVIDER` | `openai` | LLM provider backend: `openai` (OpenAI-compatible HTTP), `custom`, or `fake` |
| `CITECAT_LLM_BASE_URL` | `https://api.openai.com/v1` | Base URL for OpenAI-compatible chat completion endpoints |
| `CITECAT_LLM_MODEL` | `gpt-4o-mini` | Model identifier passed to chat completions API |
| `CITECAT_LLM_API_KEY` | `""` | Bearer token for LLM API (never logged or hardcoded) |
| `CITECAT_LLM_TIMEOUT` | `30.0` | Timeout in seconds for LLM completion requests |
| `CITECAT_LLM_MAX_RETRIES` | `2` | Maximum retry attempts on HTTP 429 and 5xx errors with exponential backoff |
| `CITECAT_LLM_MAX_TOKENS` | `1000` | Maximum output tokens generated by the LLM provider |
| `CITECAT_LLM_TEMPERATURE` | `0.0` | Sampling temperature (0.0 for deterministic factual generation) |
| `CITECAT_LLM_MAX_CONTEXT_CHARS` | `12000` | Maximum character length of evidence passages sent in LLM context |

---

## Hybrid Retrieval & Score Calibration

Raw BM25 scores ($[0, \infty)$) and cosine similarity scores ($[-1, 1]$) cannot be simply summed. CiteCat applies rigorous calibration:

1. **BM25 Lexical Calibration**:
   $$s_{\text{lex, norm}} = \frac{s_{\text{lex}}}{\max_{p \in \text{corpus}} s_{\text{lex}}(p)} \in [0, 1]$$
2. **Dense Semantic Calibration**:
   $$s_{\text{sem, norm}} = \max\left(0.0, \cos(\mathbf{q}, \mathbf{p})\right) \in [0, 1]$$
3. **Hybrid Convex Combination**:
   $$s_{\text{hybrid}} = \alpha \cdot s_{\text{lex, norm}} + (1 - \alpha) \cdot s_{\text{sem, norm}} \in [0, 1]$$

Where:
- $\alpha = 0.5$ balances keyword precision and semantic generalization.
- If the neural model fails to load or runs out of memory, the server logs a warning to `stderr` and **gracefully falls back to BM25** without crashing.

---

## Safe Embedding Storage & Staleness Detection

Embeddings are persisted directly into the SQLite database (`embeddings` table) as raw IEEE 754 32-bit float BLOBs (`struct.pack` / `struct.unpack`), avoiding external native dependencies like `sqlite-vss`:
- **Staleness Invalidation**: Each embedding row records `(record_id, start_char, end_char, text_hash, model_name, dimension)`. If a passage's text is updated or the model configuration changes, CiteCat detects the hash/dimension mismatch and regenerates the embedding.
- **Corruption Resilience**: If a vector BLOB is truncated or corrupted, CiteCat discards it gracefully and falls back to text or on-the-fly recomputation.
- **Thread Safety**: All database operations are serialized through `threading.Lock()`.

---

## Full-Text PDF Ingestion & Exact Provenance

CiteCat extends beyond abstracts by ingesting full-text open-access PDF articles directly from HAL:

1. **Open-Access Discovery**: Automatically resolves canonical PDF URLs from HAL document metadata (`fileMain_s`, `files_s`, `openAccess_bool`, and `fileLicenses_s`).
2. **Safe, Bounded Downloading**:
   - Streaming download with strict `pdf_max_size_bytes` enforcement (default 50 MB) to prevent memory exhaustion.
   - Rejects non-PDF responses (e.g. HTML error pages served under PDF URLs) and validates `%PDF-` magic header bytes.
   - Bounded redirect following (`max_redirects=5`) and exponential backoff retry.
   - SHA-256 disk caching avoiding repeated downloads of unchanged files.
3. **PyMuPDF Text Extraction & OCR Detection**:
   - Page-by-page extraction preserving 1-indexed page numbers.
   - Conservative whitespace normalization while preserving all Unicode punctuation and French accents/diacritics.
   - Scanned image detection: if a page contains raster images and fewer than 30 extractable words, CiteCat marks it as `image_only` with `ocr_required=True` instead of hallucinating garbled text.
4. **Exact Character Offset Guarantee**:
   - Extracted pages are stitched with tracked offsets such that every passage's character span satisfies:
     $$\text{full\_text}[\text{start\_char} : \text{end\_char}] == \text{passage.text}$$
   - When cited, quotations report exact page numbers and PDF source URLs (`[{title} — Page {page_number}]`).

---

## Grounded LLM Answer Synthesis (Optional)

CiteCat provides an optional LLM synthesis layer on top of the retrieval and verification pipeline:

1. **Default Mode (Extractive)**: By default (`CITECAT_SYNTHESIS_MODE="extractive"`), CiteCat returns deterministic verbatim passages with exact citation quotations. No LLM dependency, external API call, or API key is required.
2. **Provider Abstraction**:
   - `OpenAICompatibleProvider`: Built with pure `httpx` (no extra heavy SDK dependencies). Supports OpenAI, Ollama, vLLM, or any compatible endpoint. Features exponential backoff jitter on HTTP 429/5xx, timeouts, and JSON-mode structured decoding.
   - `FakeLLMProvider`: Deterministic in-memory mock provider for reproducible offline tests covering all error and edge cases.
3. **Structured Claim Grounding**:
   - The LLM receives relevant passages tagged with stable identifiers (`P1`, `P2`) and is instructed to generate structured JSON with `claims: [{claim, passage_ids, quote}]`.
   - **Fabrication Rejection**: Every cited passage ID is checked against the actual retrieved evidence set. Fabricated IDs (e.g. `P999`) are discarded.
   - **Quote Verification**: Every claim quote is verified against stored source text using `verify_citation()`. Hallucinated quotes trigger immediate claim rejection.
   - **Immutable Provenance**: Canonical HAL URLs, document licenses, and PDF page numbers are strictly copied from the local SQLite store — the LLM's generated metadata is never trusted.
   - **Conservative Claim Filtering**: If only a subset of claims verify, only verified claims are presented in the final answer. If no claims verify, the server safely abstains.
4. **Security & Prompt Injection Defenses**:
   - Evidence is strictly enclosed within `<evidence>` tags.
   - The system prompt instructs the model to treat retrieved documents as untrusted third-party data and ignore embedded instructions (e.g., `"SYSTEM OVERRIDE"`).
   - Only relevant passages up to `CITECAT_LLM_MAX_CONTEXT_CHARS` are transmitted; full PDFs are never sent to external LLMs.
   - API keys are kept in memory and never logged to stdout, stderr, or cache databases.

---

## Command-Line Interface (CLI)

CiteCat provides bounded CLI commands for offline indexing, batch processing, and server management:

```bash
# 1. Start the MCP server over stdio (default)
uv run citecat serve

# 2. Ingest a specific HAL record's full-text open-access PDF
uv run citecat ingest-pdf hal-01234567

# 3. Batch-ingest open-access records matching a search query
uv run citecat ingest-batch --query "machine learning openAccess_bool:true" --limit 10

# 4. Batch-ingest specific record IDs
uv run citecat ingest-batch --ids hal-01234567 hal-02345678
```

---

## MCP Client Configuration

CiteCat communicates over standard input/output (`stdio`). Diagnostic logging is strictly directed to `stderr` with explicit flushing to guarantee JSON-RPC output purity.

### Claude Desktop Configuration (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "citecat": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\Users\\<your-user>\\Desktop\\citeCat",
        "run",
        "citecat"
      ],
      "env": {
        "CITECAT_RETRIEVAL_MODE": "hybrid",
        "CITECAT_EMBEDDING_MODEL": "BAAI/bge-m3",
        "CITECAT_HYBRID_ALPHA": "0.5",
        "CITECAT_MIN_SIMILARITY_THRESHOLD": "0.35"
      }
    }
  }
}
```

---

## MCP Tools & Resources

### 1. `search(query: str, filters: dict | None = None, limit: int = 10) -> str`
Search HAL open-science publications. Returns structured JSON records and automatically caches responses and ingests passages for retrieval.

### 2. `get_record(id: str) -> str`
Fetch a publication by its HAL document identifier (e.g. `hal-01234567`) and ingest it into the local store.

### 3. `ingest_pdf(id: str) -> str`
Downloads and ingests the full-text open-access PDF for a HAL document:
- Resolves open-access PDF file URL (`fileMain_s` / `files_s`).
- Safe streaming download enforcing byte limits and verifying `%PDF-` header.
- Page-by-page PyMuPDF extraction preserving 1-indexed page numbers.
- Enables deep-article retrieval and page-attributed citations (`[Title — Page N]`).

### 4. `answer_with_citations(question: str, k: int = 5) -> str`
Answer a research question using ingested HAL publications with verifiable citations.
- Uses configured retrieval strategy (`bm25`, `hybrid`, or `semantic`).
- Supports cross-lingual queries (e.g., French query matching English evidence).
- Cites exact passages with page-number attribution for PDF-derived evidence.
- **Abstains cleanly** (`answer: null`, `abstention_reason: "..."`) when evidence is insufficient or question is out-of-domain.

### 5. `verify_citation(quote: str, source_url: str) -> str`
Verifies whether a claimed quotation appears verbatim or under conservative normalization (Unicode NFKD/NFD diacritics, typographic quotes, whitespace) in the source document.
- Proves literal existence in source text; does not assume semantic truth of the claim.

### 6. Resource Template: `catalogue://record/{id}`
Read record metadata and full text preview directly through MCP resource reads.

---

## Benchmark Evaluation Results

### 1. Multilingual Benchmark (Bilingual French & English)

Evaluated against bilingual French and English corpora across 14 diverse queries ([`tests/eval_dataset.py`](file:///c:/Users/windows/Desktop/citeCat/tests/eval_dataset.py)):

| Evaluation Metric | BM25-Only Baseline | Hybrid (BM25 + Semantic) |
|---|---|---|
| **Total Evaluated Questions** | 14 | 14 |
| **Answerable Questions** | 10 | 10 |
| **Unanswerable Questions** | 4 | 4 |
| **Answerable Question Accuracy** | 100.0% | **100.0%** |
| **Correct Abstention Rate** | 100.0% | **100.0%** |
| **False Positive Rate (Hallucinations)** | 0.0% | **0.0%** |
| **Mean Recall@k** | 100.0% | **100.0%** |
| **Mean Reciprocal Rank (MRR)** | 0.9500 | **1.0000** (+0.05) |

### 2. Full-Text PDF vs. Abstract-Only Benchmark

Evaluated on queries whose factual evidence resides exclusively in the body of full-text articles:

| Evaluation Metric | Abstract-Only BM25 | Full-Text BM25 | Full-Text Hybrid |
|---|---|---|---|
| **Total Evaluated Questions** | 6 | 6 | 6 |
| **Answerable (In PDF body only)** | 4 | 4 | 4 |
| **Unanswerable (Control)** | 2 | 2 | 2 |
| **Answerable Accuracy** | 0.0% (4/4 safely abstained) | **100.0%** (4/4 answered) | **100.0%** (4/4 answered) |
| **Correct Abstention Rate** | 100.0% (2/2) | **100.0%** (2/2) | **100.0%** (2/2) |
| **False Positive Rate** | 0.0% | **0.0%** | **0.0%** |
| **Recall@k** | 0.0% | **100.0%** | **100.0%** |
| **Mean Reciprocal Rank (MRR)** | 0.0000 | **1.0000** | **1.0000** |
| **Citation Verification Rate** | 100.0% | **100.0%** | **100.0%** |

### Cross-Lingual & Paraphrase Examples

1. **Cross-Lingual (FR $\rightarrow$ EN)**:
   - Query: *"Quelles sont les méthodes informatiques pour prédire la structure tridimensionnelle des protéines ?"*
   - Target: `hal-eval-en-protein` (*"Computational Methods for Protein Structure Prediction"*)
   - Result: BM25 had no vocabulary overlap; Hybrid aligned the bilingual biological concepts and retrieved the document at rank 1.
2. **Full-Text Retrieval with Page Provenance**:
   - Query: *"What depth and temperature conditions were recorded in the Southern Ocean transect?"*
   - Target: `hal-eval-pdf-marine` (*"Ocean Acidification and Marine Biodiversity in the Southern Ocean"*, Page 3)
   - Result: Abstract alone lacked transect depth/temperature; full-text retrieval extracted the exact passage from Page 3 and verified the citation quote verbatim.
3. **Safe Abstention**:
   - Query: *"What is the recipe for the best chocolate cake?"*
   - Result: Both BM25 and Hybrid scored below threshold and abstained cleanly with zero false citations.

---

### 3. Phase 5 Answer Quality & Grounded Synthesis Benchmark

Evaluated across 18 diverse queries spanning abstracts, multi-page PDFs, cross-lingual retrieval, contradictory evidence, and embedded prompt injections ([`tests/eval_phase5.py`](file:///c:/Users/windows/Desktop/citeCat/tests/eval_phase5.py)):

| Evaluation Metric | Measured Result | Benchmark Notes |
|---|:---:|---|
| **Total Evaluated Questions** | 18 | 15 answerable + 3 out-of-domain controls |
| **Answer Support Rate** | **100.0%** (15/15) | All answerable questions answered with supporting evidence |
| **Abstention Quality Rate** | **100.0%** (3/3) | 100% clean abstention on unanswerable/out-of-domain queries |
| **Citation Validity Rate** | **100.0%** | 100% of quotes verified verbatim/normalized in source text |
| **Mean Recall@k (k=5)** | **1.0000** | Target documents retrieved in top-5 for all answerable queries |
| **Mean Reciprocal Rank (MRR)** | **0.9667** | Target evidence retrieved at rank 1 in >93% of queries |
| **Prompt Injection Defense** | **100.0%** | Zero hijacking; malicious embedded payloads ignored |
| **Mean Synthesis Latency** | **2.5 ms** | Measured with deterministic in-memory provider |

> [!NOTE]
> **Limitations Disclaimer**: Sample size $N=18$ represents a deterministic regression and quality validation dataset. These metrics confirm provenance integrity, citation verification, and safety guards, but should not be interpreted as statistically significant across unbounded real-world corpora.

---

## Verification & Testing

```bash
# Run 181 offline unit, ingestion, PDF, MCP protocol, and LLM synthesis tests
uv run pytest -v

# Run live HAL integration tests (requires network access to api.archives-ouvertes.fr)
uv run pytest -m live -v

# Run opt-in live LLM integration tests (requires CITECAT_LIVE_LLM=1 and endpoint/key)
CITECAT_LIVE_LLM=1 CITECAT_LLM_API_KEY=sk-... uv run pytest -m live_llm -v

# Code quality & formatting
uv run ruff check .
uv run ruff format --check .

# Package build
uv build
```

---

## License

MIT — see [LICENSE](LICENSE).
