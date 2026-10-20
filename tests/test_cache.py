"""Tests for citecat.cache – SQLite response cache."""

from __future__ import annotations

import time

import pytest

from citecat.cache import Cache
from citecat.config import Config


@pytest.fixture
def cache_path(tmp_path):
    return str(tmp_path / "test_cache.db")


@pytest.fixture
def cache(cache_path):
    config = Config(cache_db_path=cache_path, cache_ttl_seconds=10)
    c = Cache(config)
    yield c
    c.close()


class TestCache:
    def test_put_and_get(self, cache):
        cache.put("key1", [{"id": "hal-1", "title": "Test"}])
        result = cache.get("key1")
        assert result is not None
        assert result[0]["id"] == "hal-1"

    def test_get_missing_key(self, cache):
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_expiry(self, cache_path):
        config = Config(cache_db_path=cache_path, cache_ttl_seconds=0)
        c = Cache(config)
        c.put("key1", [{"id": "hal-1"}])
        # TTL = 0 means immediately expired
        time.sleep(0.01)
        result = c.get("key1")
        assert result is None
        c.close()

    def test_cache_overwrite(self, cache):
        cache.put("key1", [{"v": 1}])
        cache.put("key1", [{"v": 2}])
        result = cache.get("key1")
        assert result[0]["v"] == 2

    def test_cache_clear(self, cache):
        cache.put("key1", [{"id": "1"}])
        cache.put("key2", [{"id": "2"}])
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_isolation(self, tmp_path):
        """Two caches with different paths don't share data."""
        c1 = Cache(Config(cache_db_path=str(tmp_path / "c1.db")))
        c2 = Cache(Config(cache_db_path=str(tmp_path / "c2.db")))
        c1.put("shared_key", [{"source": "c1"}])
        assert c2.get("shared_key") is None
        c1.close()
        c2.close()

    def test_cache_concurrent_access(self, cache):
        import concurrent.futures

        def worker(i):
            cache.put(f"key_{i}", [{"item": i}])
            return cache.get(f"key_{i}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(worker, i) for i in range(20)]
            results = [f.result() for f in futures]

        for i, res in enumerate(results):
            assert res is not None
            assert res[0]["item"] == i
