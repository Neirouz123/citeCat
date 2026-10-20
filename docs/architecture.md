# CiteCat Architecture

## Overview

CiteCat follows a layered architecture:

1. **Transport Layer** — FastMCP handles stdio/HTTP MCP protocol
2. **Tool Layer** — `server.py` exposes search, get_record, answer_with_citations, verify_citation
3. **Client Layer** — `client.py` communicates with HAL API with retries and rate limiting
4. **Cache Layer** — `cache.py` provides SQLite response caching
5. **Retrieval Layer** — `retrieval.py` implements BM25 scoring over ingested passages
6. **Citation Layer** — `citations.py` constructs verifiable citations from retrieval results
7. **Storage Layer** — `ingest.py` manages passage extraction and SQLite persistence

## Data Flow

### Search
```
MCP Client → search tool → cache check → HAL API → parse → validate → cache store → respond
```

### Answer with Citations
```
MCP Client → answer tool → load passages → BM25 retrieve → score filter → build citations → respond
```

### Citation Verification
```
MCP Client → verify tool → load source text → exact match → normalized match → respond
```

