"""Configuration management for CiteCat."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from environment variables with safe defaults."""

    # HAL API
    hal_base_url: str = "https://api.archives-ouvertes.fr"
    hal_timeout: float = 30.0
    hal_max_concurrent: int = 5
    hal_max_results: int = 100
    hal_ssl_verify: bool = True

    # Rate limiting
    rate_limit_per_second: float = 5.0

    # Retry
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0

    # Cache
    cache_db_path: str = "citecat_cache.db"
    cache_ttl_seconds: int = 3600  # 1 hour

    # Retrieval
    passage_min_words: int = 200
    passage_max_words: int = 300
    retrieval_db_path: str = "citecat_retrieval.db"
    retrieval_mode: str = "bm25"  # "bm25", "semantic", or "hybrid"
    embedding_model: str = "BAAI/bge-m3"
    hybrid_alpha: float = 0.5  # Weight for lexical vs semantic (0.0=semantic, 1.0=BM25)
    min_similarity_threshold: float = 0.35  # Threshold for semantic/hybrid relevance
    embedding_cache_dir: str = ""
    auto_embed_on_ingest: bool = True

    # PDF Retrieval & Ingestion
    pdf_timeout: float = 30.0
    pdf_max_size_bytes: int = 50 * 1024 * 1024  # 50 MB
    pdf_cache_dir: str = ""
    pdf_max_redirects: int = 5
    pdf_auto_download: bool = False
    pdf_allow_private_ips: bool = False
    pdf_max_pages: int = 1000

    # Grounded LLM Answer Synthesis (Optional)
    synthesis_mode: str = "extractive"  # "extractive" or "llm"
    llm_provider: str = "openai"  # "openai", "custom", or "fake"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_timeout: float = 30.0
    llm_max_retries: int = 2
    llm_max_tokens: int = 1000
    llm_temperature: float = 0.0
    llm_max_context_chars: int = 12000

    @classmethod
    def from_env(cls) -> Config:
        """Load configuration from environment variables.

        Environment variables are prefixed with ``CITECAT_``.  For example,
        ``CITECAT_HAL_TIMEOUT=60`` overrides the default timeout.
        """
        kwargs: dict = {}
        for f in cls.__dataclass_fields__:
            env_key = f"CITECAT_{f.upper()}"
            val = os.environ.get(env_key)
            if val is not None:
                target_type = cls.__dataclass_fields__[f].type
                if target_type in ("float", float):
                    kwargs[f] = float(val)
                elif target_type in ("int", int):
                    kwargs[f] = int(val)
                elif target_type in ("bool", bool):
                    kwargs[f] = val.lower() in ("true", "1", "yes")
                else:
                    kwargs[f] = val
        return cls(**kwargs)
