# CiteCat

CiteCat is a Model Context Protocol (MCP) server for academic research using [HAL](https://hal.science/), France's open-science national archive. It enables AI assistants and researchers to discover scientific publications, ingest open-access full-text PDFs, and produce answers grounded in source material with verifiable, page-level citations.

---

## Key Features

- **Academic Search & Record Retrieval**: Queries the official HAL Solr API with support for structured filters, pagination, concurrency controls, rate limiting, and local response caching.
- **Hybrid Lexical & Dense Retrieval**: Combines Okapi BM25 keyword matching with multilingual dense vector search (such as `BAAI/bge-m3` or deterministic offline models) using calibrated score normalization for cross-lingual (French $\leftrightarrow$ English) search.
- **Full-Text PDF Ingestion & Page-Level Provenance**: Streams open-access PDFs with strict byte limits, SSRF safeguards, and magic-byte checks; extracts text page-by-page via PyMuPDF; and tracks exact character offsets with 1-indexed page attribution (`[Title — Page N]`).
- **Citation Verification & Safe Abstention**: Verifies quotations verbatim and under conservative normalization (handling accents, diacritics, typographic quotes, and ligatures). Safely abstains with a clear reason when evidence is insufficient, preventing factual fabrication.
- **Optional Grounded LLM Synthesis**: Supports structured synthesis via OpenAI-compatible endpoints (including Google Gemini, local Ollama, or vLLM). Every generated claim is validated against retrieved passages, and citations use stored document metadata rather than model claims.
- **Resilient Network Handling & Security Protections**: Thread-safe SQLite storage, exponential backoff with jitter on transient HTTP errors, prompt-injection defense with delimited `<evidence>` blocks, and strict credential isolation (API keys are never logged or stored in output).

---

## Why CiteCat?

Most AI research tools behave like generic conversational chatbots—summarizing papers from parametric memory, fabricating plausible-sounding references, or generating ungrounded claims.

CiteCat takes a research-first approach focused on traceability and reproducibility:

- **Source Grounding**: Every asserted finding must link to an ingested document passage.
- **Traceable Attribution**: PDF-derived evidence points directly to specific page numbers and canonical HAL URLs.
- **Honest Abstention**: When retrieved publications lack sufficient evidence to answer a question, CiteCat abstains explicitly instead of guessing.
- **Protocol Purity**: Communicates over standard MCP (`stdio`), keeping diagnostic logs on `stderr` to guarantee pure JSON-RPC transport for MCP clients.

---

## Requirements

- Python 3.11 or later
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

---

## Quick Start

### 1. Clone the repository

```powershell
# Windows PowerShell
git clone https://github.com/Neirouz123/citeCat.git
cd citeCat
```

```bash
# Linux / macOS
git clone https://github.com/Neirouz123/citeCat.git
cd citeCat
```

### 2. Install dependencies

Choose the installation tier that fits your use case:

```powershell
# Core installation: BM25 lexical retrieval, lightweight, no ML dependencies
uv sync

# Add full-text PDF ingestion support (PyMuPDF)
uv sync --extra pdf

# Full installation: PDF ingestion, multilingual neural embeddings, and dev tools
uv sync --extra all
```

*(Using standard `pip`: `pip install -e .` or `pip install -e ".[all]"`).*

### 3. Minimum Configuration

No configuration is required to get started. By default, CiteCat runs in deterministic extractive mode using BM25 lexical retrieval and the public HAL API.

### 4. Run the MCP Server

```powershell
uv run citecat
```

Alternatively, run via Python module:

```powershell
uv run python -m citecat.server
```

---

## MCP Client Configuration

To connect CiteCat to an MCP-compatible client (such as Claude Desktop or Cursor), add the server entry to your client configuration:

### Configuration Example (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "citecat": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\path\\to\\citeCat",
        "run",
        "citecat"
      ]
    }
  }
}
```

*(On Linux or macOS, replace `"C:\\path\\to\\citeCat"` with your absolute path, e.g. `"/home/user/citeCat"`).*

---

## Optional LLM Answer Synthesis

By default, CiteCat operates in deterministic `extractive` mode, returning verified passages directly without calling external language models.

To enable generative synthesis grounded in verified passages, set the synthesis mode and configure an OpenAI-compatible provider:

### Google Gemini (OpenAI-Compatible Endpoint)

```powershell
# Windows PowerShell
$env:CITECAT_SYNTHESIS_MODE = "llm"
$env:CITECAT_LLM_PROVIDER = "openai"
$env:CITECAT_LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
$env:CITECAT_LLM_MODEL = "gemini-flash-lite-latest"
$env:CITECAT_LLM_API_KEY = "your-api-key-here"
uv run citecat
```

```bash
# Linux / macOS
export CITECAT_SYNTHESIS_MODE="llm"
export CITECAT_LLM_PROVIDER="openai"
export CITECAT_LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export CITECAT_LLM_MODEL="gemini-flash-lite-latest"
export CITECAT_LLM_API_KEY="your-api-key-here"
uv run citecat
```

### Local Ollama (Private, Offline)

```bash
export CITECAT_SYNTHESIS_MODE="llm"
export CITECAT_LLM_BASE_URL="http://localhost:11434/v1"
export CITECAT_LLM_MODEL="llama3.1:8b"
export CITECAT_LLM_API_KEY="ollama"
uv run citecat
```

---

## Testing

CiteCat comes with a comprehensive test suite covering HAL client integration, caching, PDF extraction, hybrid ranking, citation verification, and MCP protocol compliance:

```powershell
# Run the complete offline test suite (187 tests, 100% offline)
uv run pytest

# Check code linting and style
uv run ruff check .

# Check code formatting
uv run ruff format --check .

# Build package distributions (wheel and sdist)
uv build
```

---

## Limitations & Notes

- **Source Scope**: CiteCat searches open-access publications deposited on HAL. If a record does not include an open-access PDF deposit or an abstract, full-text retrieval may not be available.
- **Critical Evaluation**: Citation verification confirms that quoted passages exist verbatim in the source documents; it does not replace researcher evaluation of scientific methodology or peer review status.
- **Scanned Documents**: Scanned image-only PDFs without a text layer are detected and flagged (`ocr_required=True`) rather than generating ungrounded text.

---

## License

This project is licensed under the [MIT License](LICENSE).

