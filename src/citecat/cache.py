"""SQLite response cache with configurable TTL."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from typing import Any

from citecat.config import Config

logger = logging.getLogger(__name__)


class Cache:
    """Thread-safe SQLite cache for HAL API responses.

    Stores serialised JSON responses keyed by a stable request hash.
    Expired entries are lazily purged on reads.
    """

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Config()
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                self._config.cache_db_path,
                check_same_thread=False,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS response_cache (
                    key       TEXT PRIMARY KEY,
                    response  TEXT NOT NULL,
                    stored_at REAL NOT NULL
                )
                """
            )
            self._conn.commit()
        return self._conn

    def get(self, key: str) -> list[dict[str, Any]] | None:
        """Return cached response data if present and not expired."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT response, stored_at FROM response_cache WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            response_json, stored_at = row
            if time.time() - stored_at > self._config.cache_ttl_seconds:
                conn.execute("DELETE FROM response_cache WHERE key = ?", (key,))
                conn.commit()
                return None
            try:
                return json.loads(response_json)
            except (json.JSONDecodeError, ValueError):
                logger.warning("Corrupt cache entry for key %s, removing", key)
                conn.execute("DELETE FROM response_cache WHERE key = ?", (key,))
                conn.commit()
                return None

    def put(self, key: str, data: Any) -> None:
        """Store response data in the cache."""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                INSERT OR REPLACE INTO response_cache (key, response, stored_at)
                VALUES (?, ?, ?)
                """,
                (key, json.dumps(data, default=str), time.time()),
            )
            conn.commit()

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM response_cache")
            conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
