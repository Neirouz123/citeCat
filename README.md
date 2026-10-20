# CiteCat

CiteCat is a Model Context Protocol ([MCP](https://modelcontextprotocol.io/)) server for academic research using [HAL](https://hal.science/), France's open-science national repository. It enables AI assistants and researchers to search scientific publications, ingest open-access full-text PDFs, and produce research answers grounded in source material with verifiable, page-level citations.

---

## Features

- **Academic Search & Record Retrieval**: Queries the official HAL Solr API with support for structured Solr filter queries (`docType_s`, publication years, domains), pagination, rate limiting, concurrency bounds, and local SQLite caching.
- **Hybrid Lexical & Dense Retrieval**: Combines Okapi BM25 keyword matching with optional multilingual dense vector embeddings (such as `BAAI/bge-m3` via `sentence-transformers` or a lightweight offline model) using calibrated score normalization for cross-lingual (French $\leftrightarrow$ English) search.
- **Full-Text PDF Ingestion**: Downloads open-access PDFs with strict byte limits, redirect bounds, magic-byte (`%PDF-`) validation, and Server-Side Request Forgery (SSRF) protections. Extracts text page-by-page via PyMuPDF while tracking character spans and page numbers (`[Title — Page N]`).
- **Citation Verification & Safe Abstention**: Verifies quotations verbatim and under conservative normalization (Unicode NFKC, diacritics stripping, typographic quotes, and French ligatures). Safely abstains with a clear reason when evidence is insufficient, preventing ungrounded assertions.
- **Optional Grounded LLM Synthesis**: Supports structured synthesis via OpenAI-compatible endpoints (including Google Gemini, local Ollama, or vLLM). Every generated claim is validated against retrieved evidence, and citations use stored document metadata rather than model claims.
- **Protocol Purity & Security**: Operates over standard MCP `stdio`. Diagnostic logging is strictly directed to `stderr` with explicit flushing to protect JSON-RPC transport integrity. Credentials and authorization headers are never logged or exposed in output payloads.

---

## Requirements

- Python 3.11 or later
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

---

## Installation

### Installation using `uv` (Recommended)

CiteCat provides tiered installation options depending on your research needs:

```bash
# Clone the repository
git clone https://github.com/Neirouz123/citeCat.git
cd citeCat
```

#### 1. Core Installation (BM25 Lexical Baseline)
Lightweight installation with zero machine-learning dependencies:
```bash
uv sync
```

#### 2. Development & Testing
Includes test runners, linting tools, and PDF parsing dependencies:
```bash
uv sync --extra dev
```

#### 3. PDF Full-Text Support
Enables open-access PDF downloading and PyMuPDF text extraction:
```bash
uv sync --extra pdf
```

#### 4. Semantic Search Support
Enables multilingual dense embeddings (`sentence-transformers` and `numpy`):
```bash
uv sync --extra semantic
```

#### 5. Full Installation
Installs all optional dependencies (development, semantic search, and PDF ingestion):
```bash
uv sync --extra all
```

---

### Installation using `pip`

CiteCat can also be installed using standard `pip`:

```bash
# Core installation
pip install .

# With PDF support
pip install ".[pdf]"

# With semantic retrieval
pip install ".[semantic]"

# Full installation with all extras
pip install ".[all]"

# Editable development installation
pip install -e ".[dev]"
```

---

## MCP Client Configuration

CiteCat communicates over standard input/output (`stdio`). Add CiteCat to your MCP client configuration file:

### Claude Desktop Configuration (`claude_desktop_config.json`)

#### Windows PowerShell Path Example:
```json
{
  "mcpServers": {
    "citecat": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\Users\\username\\Desktop\\citeCat",
        "run",
        "citecat"
      ]
    }
  }
}
```

#### Linux / macOS Path Example:
```json
{
  "mcpServers": {
    "citecat": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/username/citeCat",
        "run",
        "citecat"
      ]
    }
  }
}
```

### Running the Server Directly

You can test the server directly from the command line:

```bash
# Using the installed script entry point
uv run citecat

# Alternatively, using the Python module syntax
uv run python -m citecat.server
```

---

## CLI Ingestion Commands

CiteCat provides CLI subcommands for downloading and indexing papers outside of interactive MCP sessions:

```bash
# Ingest an open-access PDF for a specific HAL document ID
uv run citecat ingest-pdf hal-01234567

# Batch ingest records and PDFs matching a search query
uv run citecat ingest-batch --query "machine learning" --limit 5

# Batch ingest specific record IDs
uv run citecat ingest-batch --ids "hal-01234567,hal-02345678" --limit 2
```

---

## Optional LLM Provider Configuration

By default, CiteCat operates in deterministic `extractive` mode (`CITECAT_SYNTHESIS_MODE="extractive"`), returning verbatim passages without calling external language models.

To enable grounded generative synthesis, configure an OpenAI-compatible endpoint:

### Google Gemini (OpenAI-Compatible Endpoint)

CiteCat automatically normalizes Google Gemini endpoints and selects appropriate high-throughput models (`gemini-flash-lite-latest`):

#### PowerShell (Windows):
```powershell
$env:CITECAT_SYNTHESIS_MODE = "llm"
$env:CITECAT_LLM_PROVIDER = "openai"
$env:CITECAT_LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
$env:CITECAT_LLM_MODEL = "gemini-flash-lite-latest"
$env:CITECAT_LLM_API_KEY = "your-gemini-api-key"
uv run citecat
```

#### Bash (Linux / macOS):
```bash
export CITECAT_SYNTHESIS_MODE="llm"
export CITECAT_LLM_PROVIDER="openai"
export CITECAT_LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export CITECAT_LLM_MODEL="gemini-flash-lite-latest"
export CITECAT_LLM_API_KEY="your-gemini-api-key"
uv run citecat
```

---

## Local Ollama Configuration

You can run CiteCat with a local [Ollama](https://ollama.com/) instance for completely offline, private synthesis without sending data to external cloud providers:

#### 1. Start Ollama and download a model:
```bash
ollama run llama3.1:8b
```

#### 2. Configure CiteCat to use the local endpoint:

#### PowerShell (Windows):
```powershell
$env:CITECAT_SYNTHESIS_MODE = "llm"
$env:CITECAT_LLM_PROVIDER = "openai"
$env:CITECAT_LLM_BASE_URL = "http://localhost:11434/v1"
$env:CITECAT_LLM_MODEL = "llama3.1:8b"
$env:CITECAT_LLM_API_KEY = "ollama"
uv run citecat
```

#### Bash (Linux / macOS):
```bash
export CITECAT_SYNTHESIS_MODE="llm"
export CITECAT_LLM_PROVIDER="openai"
export CITECAT_LLM_BASE_URL="http://localhost:11434/v1"
export CITECAT_LLM_MODEL="llama3.1:8b"
export CITECAT_LLM_API_KEY="ollama"
uv run citecat
```

---

## Configuration Reference

All settings are configured through environment variables prefixed with `CITECAT_`:

| Environment Variable | Default | Description |
|---|---|---|
| `CITECAT_RETRIEVAL_MODE` | `bm25` | Retrieval strategy: `bm25`, `hybrid`, or `semantic` |
| `CITECAT_EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model name (`sentence-transformers` model or `mock-multilingual-v1`) |
| `CITECAT_HYBRID_ALPHA` | `0.5` | Weight for lexical vs. semantic scoring ($0.0$ = dense only, $1.0$ = BM25 only) |
| `CITECAT_MIN_SIMILARITY_THRESHOLD` | `0.35` | Minimum relevance score for semantic and hybrid retrieval |
| `CITECAT_HAL_BASE_URL` | `https://api.archives-ouvertes.fr` | Official HAL Solr API base URL |
| `CITECAT_HAL_TIMEOUT` | `30.0` | Timeout in seconds for HAL API requests |
| `CITECAT_HAL_MAX_CONCURRENT` | `5` | Maximum concurrent requests to HAL API |
| `CITECAT_RATE_LIMIT_PER_SECOND` | `5.0` | Maximum requests per second to HAL API |
| `CITECAT_MAX_RETRIES` | `3` | Maximum retry attempts for transient HTTP errors |
| `CITECAT_CACHE_DB_PATH` | `citecat_cache.db` | SQLite cache database path |
| `CITECAT_CACHE_TTL_SECONDS` | `3600` | Cache time-to-live in seconds (default 1 hour) |
| `CITECAT_RETRIEVAL_DB_PATH` | `citecat_retrieval.db` | SQLite retrieval store database path |
| `CITECAT_HAL_SSL_VERIFY` | `true` | Verify SSL certificates for HTTP requests |
| `CITECAT_PDF_TIMEOUT` | `30.0` | Timeout in seconds for PDF downloads |
| `CITECAT_PDF_MAX_SIZE_BYTES` | `52428800` | Maximum PDF download size in bytes (default 50 MB) |
| `CITECAT_PDF_MAX_REDIRECTS` | `5` | Maximum HTTP redirect hops allowed for PDF downloads |
| `CITECAT_PDF_ALLOW_PRIVATE_IPS` | `false` | Allow downloads from private IP ranges (strictly `false` for SSRF defense) |
| `CITECAT_PDF_MAX_PAGES` | `1000` | Maximum pages processed per document |
| `CITECAT_SYNTHESIS_MODE` | `extractive` | Synthesis mode: `extractive` (deterministic verbatim) or `llm` |
| `CITECAT_LLM_PROVIDER` | `openai` | LLM backend: `openai`, `custom`, or `fake` (testing) |
| `CITECAT_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible chat completions base URL |
| `CITECAT_LLM_MODEL` | `gpt-4o-mini` | Model identifier passed in chat completions payload |
| `CITECAT_LLM_API_KEY` | `""` | Bearer token for LLM authentication |
| `CITECAT_LLM_TIMEOUT` | `30.0` | Timeout in seconds for LLM completion requests |
| `CITECAT_LLM_MAX_RETRIES` | `2` | Maximum retry attempts on 429 and 5xx errors |
| `CITECAT_LLM_TEMPERATURE` | `0.0` | Sampling temperature for factual synthesis |
| `CITECAT_LLM_MAX_CONTEXT_CHARS` | `12000` | Maximum character length of evidence passages sent to the LLM |

---

## Retrieval & Citation Workflow

CiteCat implements a five-stage grounded research workflow:

```
1. Search & Ingest ──► 2. Passage Ranking ──► 3. Score Filtering ──► 4. Grounded Synthesis ──► 5. Provenance Output
   (HAL API / PDF)       (BM25 / Hybrid)        (Threshold Check)       (Extractive / LLM)       (Verified Citations)
```

1. **Search & Ingestion**:
   - `search`: Queries the HAL API and caches records in SQLite (`citecat_cache.db`). Abstracts are split into passages and indexed into the retrieval store (`citecat_retrieval.db`).
   - `ingest_pdf`: Downloads open-access PDFs, extracts text page-by-page preserving page numbers, and indexes passages with exact character offsets.
2. **Passage Ranking**:
   - In default lexical mode, passages are scored using Okapi BM25 with French and English stop-word filtering.
   - In hybrid mode (`CITECAT_RETRIEVAL_MODE="hybrid"`), lexical BM25 scores and dense cosine similarities are normalized into $[0, 1]$ and combined:
     $$s_{\text{hybrid}} = \alpha \cdot s_{\text{lex, norm}} + (1 - \alpha) \cdot s_{\text{sem, norm}}$$
3. **Filtering & Safe Abstention**:
   - If retrieved passages score below the relevance threshold (`min_score`), CiteCat explicitly abstains (`answer: null`, with a clear explanation) instead of fabricating an answer.
4. **Answer Generation**:
   - **Extractive Mode (Default)**: Formats verbatim passage snippets with their titles, page numbers, and source URLs.
   - **LLM Mode**: The model generates claims referencing specific passage identifiers (`P1`, `P2`) and quotation snippets.
5. **Provenance Verification**:
   - Every quoted passage is verified against the stored source text using `verify_citation()`. Claims with unverified quotes or invalid passage IDs are discarded.
   - All source URLs, licenses, and page numbers are drawn directly from the local SQLite database, never from model-generated text.

---

## MCP Tools & Resources Reference

CiteCat exposes the following tools to connected MCP clients:

| Tool / Resource | Description |
|---|---|
| `search(query, filters, limit)` | Search HAL scientific publications by free text and Solr filters. Ingests records into the local store. |
| `get_record(id)` | Fetch a publication by its HAL document ID (e.g. `hal-01234567`) and ingest it for retrieval. |
| `ingest_pdf(id)` | Download and parse the open-access PDF for a HAL document, extracting text with page-level tracking. |
| `answer_with_citations(question, k)` | Answer a research question using ingested publications, returning verified citations and abstaining if evidence is insufficient. |
| `verify_citation(quote, source_url)` | Verify whether a claimed quotation appears verbatim or under conservative normalization in stored source text. |
| `catalogue://record/{id}` | MCP Resource template providing JSON metadata and full-text previews for ingested HAL publications. |

---

## Testing & Development

CiteCat includes a comprehensive test suite covering client handling, caching, PDF extraction, hybrid ranking, citation verification, and MCP protocol compliance.

```bash
# Run the complete offline test suite (187 tests, zero network access)
uv run pytest

# Check code linting and style
uv run ruff check .

# Check code formatting
uv run ruff format --check .

# Build package distributions (sdist and wheel)
uv build
```

### Live Network Smoke Tests (Optional)

Tests marked with `@pytest.mark.live` verify end-to-end communication with the live HAL API:

```bash
uv run pytest -m live -v
```

---

## Optional Live LLM Acceptance Tests

CiteCat includes integration tests in `tests/test_live_llm.py` marked with `@pytest.mark.live_llm` to validate real LLM providers across multilingual, PDF full-text, and prompt-injection scenarios.

These tests are **opt-in** and excluded from standard test runs.

### Prerequisites

Set `CITECAT_LIVE_LLM=1` and configure your API key or local endpoint:

#### With Google Gemini:
```powershell
# PowerShell
$env:CITECAT_LIVE_LLM = "1"
$env:CITECAT_LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
$env:CITECAT_LLM_MODEL = "gemini-flash-lite-latest"
$env:CITECAT_LLM_API_KEY = "your-gemini-api-key"
uv run pytest -m live_llm -v
```

```bash
# Bash
export CITECAT_LIVE_LLM="1"
export CITECAT_LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export CITECAT_LLM_MODEL="gemini-flash-lite-latest"
export CITECAT_LLM_API_KEY="your-gemini-api-key"
uv run pytest -m live_llm -v
```

#### With Local Ollama:
```bash
export CITECAT_LIVE_LLM="1"
export CITECAT_LLM_BASE_URL="http://localhost:11434/v1"
export CITECAT_LLM_MODEL="llama3.1:8b"
export CITECAT_LLM_API_KEY="ollama"
uv run pytest -m live_llm -v
```

---

## Limitations & Security Considerations

- **Source Scope**: CiteCat searches documents deposited on [HAL](https://hal.science/). Papers without an open-access PDF deposit or abstract cannot be searched for full-text body passages.
- **Verification Boundaries**: Citation verification confirms that a quoted phrase exists in the source text; it does not independently assess the scientific validity or peer-review quality of the publication. Researchers should verify cited findings in context.
- **Scanned Documents**: Image-only PDFs without an embedded OCR text layer are flagged (`ocr_required=True`) rather than hallucinating text.
- **SSRF & Download Safeguards**: The PDF downloader rejects private, link-local, and loopback IP addresses, enforces redirect limits (default 5), and streams files with strict byte limits (default 50 MB) to prevent resource exhaustion.
- **Prompt Injection Defense**: Retrieved evidence is isolated within `<evidence>` delimiters, and the grounding system prompt explicitly instructs the synthesizer to disregard instructions embedded in untrusted scientific articles.
- **Credential Hygiene**: API keys and authorization headers are never logged to console streams or written to SQLite databases.

---

## Contributing

Contributions are welcome. Before opening a pull request:

1. Ensure all offline tests pass: `uv run pytest`
2. Ensure code is formatted and passes linting: `uv run ruff check .` and `uv run ruff format --check .`
3. Verify that new features maintain offline determinism and do not introduce hardcoded credentials or required network calls in default test paths.

---

## License

This project is licensed under the [MIT License](LICENSE).
