"""Tests for the Redis cache client (blueprint sections 4.2 + 13).

Assumed interface
-----------------
``cache/redis_client.py`` provides get / set / invalidate over a Redis backend,
keyed by domain. The blueprint specifies:

* Key format: ``adis:cache:<domain>``            (section 4.2)
* TTL 1 hour (3600s) for safe results            (section 4.1)
* TTL 15 minutes (900s) for suspicious/malicious (section 4.1)

Because exact function names aren't fixed by the blueprint, the tests probe a
small set of conventional names (e.g. ``get_cached_result`` /
``set_cached_result`` / ``invalidate``) and skip cleanly if none are present.

The backend is swapped for an in-memory fake (``conftest.FakeRedis``) so no
real Redis/Upstash instance is required.
"""

from __future__ import annotations

import json

import pytest

from conftest import first_attr_or_skip, import_or_skip

DOMAIN = "secure-login-paypa1.xyz"
EXPECTED_KEY = f"adis:cache:{DOMAIN}"

SAFE_TTL = 3600
RISKY_TTL = 900


@pytest.fixture
def cache_module():
    return import_or_skip("cache.redis_client")


@pytest.fixture
def wired_cache(cache_module, fake_redis, monkeypatch):
    """Point the cache module at the in-memory fake Redis, however it's wired.

    Handles the common patterns: a module-level ``redis_client`` / ``client`` /
    ``r`` singleton, or a ``get_client()`` factory.
    """
    injected = False
    for attr in ("redis_client", "client", "r", "_redis", "redis_conn"):
        if hasattr(cache_module, attr):
            monkeypatch.setattr(cache_module, attr, fake_redis, raising=False)
            injected = True
    for factory in ("get_client", "get_redis", "get_redis_client"):
        if hasattr(cache_module, factory):
            monkeypatch.setattr(cache_module, factory, lambda *a, **k: fake_redis, raising=False)
            injected = True
    if not injected:
        pytest.skip("Could not locate a Redis client seam to inject the fake into.")
    return cache_module, fake_redis


def _call_set(module, domain, result, label=None):
    setter = first_attr_or_skip(
        module, ["set_cached_result", "cache_result", "set_result", "set", "put"]
    )
    # Try a few common signatures.
    for kwargs in ({"label": label}, {}):
        try:
            return setter(domain, result, **{k: v for k, v in kwargs.items() if v is not None})
        except TypeError:
            continue
    return setter(domain, result)


def _call_get(module, domain):
    getter = first_attr_or_skip(
        module, ["get_cached_result", "get_result", "get_cached", "get"]
    )
    return getter(domain)


# ---------------------------------------------------------------------------
# Key format.
# ---------------------------------------------------------------------------
def test_cache_key_format(cache_module):
    """If a key builder is exposed, it must produce ``adis:cache:<domain>``."""
    builder = None
    for name in ("cache_key", "make_key", "build_key", "_key", "_cache_key"):
        builder = getattr(cache_module, name, None)
        if builder is not None:
            break
    if builder is None:
        pytest.skip("no standalone cache-key builder exposed.")
    assert builder(DOMAIN) == EXPECTED_KEY


# ---------------------------------------------------------------------------
# Roundtrip.
# ---------------------------------------------------------------------------
def test_set_then_get_roundtrip(wired_cache):
    module, backend = wired_cache
    payload = {"domain": DOMAIN, "score": 0.93, "label": "malicious", "reasons": ["x"]}

    _call_set(module, DOMAIN, payload, label="malicious")
    fetched = _call_get(module, DOMAIN)

    # The client may return the dict directly or a JSON string — accept both.
    if isinstance(fetched, str):
        fetched = json.loads(fetched)
    assert fetched is not None
    assert fetched.get("domain") == DOMAIN
    assert fetched.get("label") == "malicious"


def test_get_miss_returns_none(wired_cache):
    module, _ = wired_cache
    assert _call_get(module, "never-cached-domain.test") in (None, {}, "")


def test_stored_under_namespaced_key(wired_cache):
    module, backend = wired_cache
    _call_set(module, DOMAIN, {"domain": DOMAIN, "label": "safe"}, label="safe")
    # The fake backend should now hold exactly the blueprint's key.
    assert EXPECTED_KEY in backend.store, (
        f"expected cache key {EXPECTED_KEY!r}, found {list(backend.store)}"
    )


# ---------------------------------------------------------------------------
# TTL tiers.
# ---------------------------------------------------------------------------
def test_safe_result_ttl_is_one_hour(wired_cache):
    module, backend = wired_cache
    _call_set(module, "github.com", {"domain": "github.com", "label": "safe"}, label="safe")
    key = "adis:cache:github.com"
    if key not in backend.ttls:
        pytest.skip("cache set did not record a TTL on the fake backend.")
    assert backend.ttls[key] == SAFE_TTL


def test_risky_result_ttl_is_fifteen_minutes(wired_cache):
    module, backend = wired_cache
    _call_set(module, DOMAIN, {"domain": DOMAIN, "label": "malicious"}, label="malicious")
    if EXPECTED_KEY not in backend.ttls:
        pytest.skip("cache set did not record a TTL on the fake backend.")
    assert backend.ttls[EXPECTED_KEY] == RISKY_TTL


# ---------------------------------------------------------------------------
# Invalidate.
# ---------------------------------------------------------------------------
def test_invalidate_removes_entry(wired_cache):
    module, backend = wired_cache
    _call_set(module, DOMAIN, {"domain": DOMAIN, "label": "safe"}, label="safe")

    invalidate = None
    for name in ("invalidate", "invalidate_domain", "delete", "evict", "clear_domain"):
        invalidate = getattr(module, name, None)
        if invalidate is not None:
            break
    if invalidate is None:
        pytest.skip("no invalidate/delete function exposed.")

    invalidate(DOMAIN)
    assert EXPECTED_KEY not in backend.store


# ---------------------------------------------------------------------------
# Resilience: a backend outage must not crash callers (cache is best-effort).
# ---------------------------------------------------------------------------
def test_get_survives_backend_error(cache_module, monkeypatch):
    class _BrokenRedis:
        def get(self, *a, **k):
            raise ConnectionError("redis down")

        def __getattr__(self, _):
            def _boom(*a, **k):
                raise ConnectionError("redis down")
            return _boom

    injected = False
    for attr in ("redis_client", "client", "r", "_redis", "redis_conn"):
        if hasattr(cache_module, attr):
            monkeypatch.setattr(cache_module, attr, _BrokenRedis(), raising=False)
            injected = True
    if not injected:
        pytest.skip("no Redis client seam to inject a broken backend into.")

    getter = getattr(cache_module, "get_cached_result", None) or getattr(
        cache_module, "get_result", None
    )
    if getter is None:
        pytest.skip("no getter exposed to test resilience.")

    # A cache-miss-on-error is acceptable; an unhandled exception is not.
    try:
        result = getter(DOMAIN)
    except ConnectionError:
        pytest.fail("cache getter should swallow backend errors and treat as a miss")
    assert result in (None, {}, "")
