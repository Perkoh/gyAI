"""
cache/redis_client.py
─────────────────────
Redis caching layer for ADIS.

Responsibilities
────────────────
  • Domain analysis result cache — main hot path
      get(domain)                  → cached result dict  (cache HIT, <5ms)
      set(domain, result, label)   → store with label-driven TTL
      delete(domain)               → invalidate single domain
      exists(domain)               → O(1) presence check without fetching

  • Network-feature sub-caches — FLAG 4 (WHOIS rate-limit mitigation)
      get_whois_features(domain)         → cached WHOIS dict | None
      set_whois_features(domain, data)   → 24h TTL
      get_dns_features(domain)           → cached DNS dict | None
      set_dns_features(domain, data)     → 24h TTL

  • Admin operations (POST /admin/cache/flush)
      flush_domain_cache()    → SCAN+DEL adis:cache:*  → count deleted
      flush_network_cache()   → SCAN+DEL adis:whois:* + adis:dns:*
      flush_all()             → SCAN+DEL adis:*        → count deleted
      flush_domain(domain)    → delete all 3 keys for one domain

  • Health & diagnostics (GET /health, GET /stats)
      health_check()          → bool   (PING round-trip)
      get_stats()             → CacheStats dataclass
      ttl_remaining(domain)   → seconds left on a domain key

  • Bulk operations (POST /analyze/bulk)
      get_multi(domains)      → Dict[domain, result|None]  (1 × MGET)
      set_multi(entries)      → int written                (1 × PIPELINE)

Design principles
─────────────────
  • Lazy init — Redis client created on first use, not at import time.
    Essential for Gunicorn: workers forked from master must each create
    their own connection pools post-fork.

  • Graceful degradation — every public method catches all Redis/network
    errors, logs a warning, and returns a safe sentinel (None / False /
    0 / empty CacheStats).  A Redis outage is always a cache miss, never
    a 500 error for the API caller.

  • decode_responses=True — client always yields str, never bytes.
    Values are JSON-serialised; no pickle, no binary blobs.

  • SCAN-based flush — flush helpers use SCAN + batched DEL (100-key
    windows) rather than FLUSHDB.  FLUSHDB would erase flask-limiter
    rate-limit keys and any other tenants sharing the Upstash instance.

  • Upstash compatibility — rediss:// URLs (TLS), socket timeouts, and
    ssl_cert_reqs="none" for managed certificates are handled auto.

TTL reference
─────────────
  safe domain result   :   3,600 s  (1 hour)   — settings.REDIS_TTL_SAFE
  suspicious result    :     900 s  (15 min)   — settings.REDIS_TTL_SUSPICIOUS
  malicious result     :     900 s  (15 min)   — settings.REDIS_TTL_MALICIOUS
  WHOIS features       :  86,400 s  (24 hours) — FLAG 4
  DNS features         :  86,400 s  (24 hours) — FLAG 4

Key namespaces (from config.constants.CacheNamespace)
──────────────────────────────────────────────────────
  adis:cache:<domain>   — analysis result cache
  adis:whois:<domain>   — WHOIS feature cache
  adis:dns:<domain>     — DNS feature cache
  adis:ratelimit:*      — flask-limiter (never touched here)
  adis:admin:lock       — model-reload lock (never touched here)

Usage
─────
    from cache.redis_client import cache

    # Analysis result cache
    result = cache.get("evil.xyz")                      # None on miss
    ok     = cache.set("evil.xyz", result_dict, "malicious")
    ok     = cache.delete("evil.xyz")
    alive  = cache.exists("evil.xyz")

    # Network sub-caches (used by features/network.py)
    whois  = cache.get_whois_features("evil.xyz")       # None on miss
    ok     = cache.set_whois_features("evil.xyz", whois_dict)
    dns    = cache.get_dns_features("evil.xyz")
    ok     = cache.set_dns_features("evil.xyz", dns_dict)

    # Bulk (for /analyze/bulk endpoint)
    hits   = cache.get_multi(["a.com", "b.com", "c.com"])   # 1 MGET
    n      = cache.set_multi([("a.com", r1, "safe"), ...])   # 1 pipeline

    # Admin
    deleted = cache.flush_domain_cache()
    deleted = cache.flush_all()
    n       = cache.flush_domain("evil.xyz")

    # Health / stats
    alive = cache.health_check()
    stats = cache.get_stats()       # CacheStats
    secs  = cache.ttl_remaining("evil.xyz")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# ---------------------------------------------------------------------------
# Guard: redis SDK import
# ---------------------------------------------------------------------------
try:
    import redis as _redis_pkg
    from redis import Redis
    from redis.exceptions import (
        ConnectionError as RedisConnectionError,
        TimeoutError as RedisTimeoutError,
        ResponseError as RedisResponseError,
        RedisError,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'redis' package is not installed.  "
        "Run: pip install redis==5.0.4"
    ) from exc

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from config.settings import settings
from config.constants import CacheNamespace


# ══════════════════════════════════════════════════════════════════════════════
# Module-level constants
# ══════════════════════════════════════════════════════════════════════════════

# FLAG 4: cache network lookups for 24 hours to avoid WHOIS rate bans
_NETWORK_CACHE_TTL: int = 86_400   # 24 hours in seconds

# Keys fetched per SCAN iteration (balance between round-trips and memory)
_SCAN_COUNT: int = 100

# Maximum keys deleted per DEL command (avoid oversized payloads)
_DEL_BATCH_SIZE: int = 500


# ══════════════════════════════════════════════════════════════════════════════
# Return types
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CacheStats:
    """Diagnostic snapshot returned by :meth:`RedisClient.get_stats`."""
    domain_cache_keys: int = 0       # keys matching adis:cache:*
    whois_cache_keys: int  = 0       # keys matching adis:whois:*
    dns_cache_keys: int    = 0       # keys matching adis:dns:*
    total_adis_keys: int   = 0       # all adis:* keys
    used_memory_human: str = "unknown"
    redis_version: str     = "unknown"
    ping_ms: float         = -1.0    # PING round-trip in milliseconds
    connected: bool        = False


# ══════════════════════════════════════════════════════════════════════════════
# Main client
# ══════════════════════════════════════════════════════════════════════════════

class RedisClient:
    """
    Fault-tolerant Redis caching client for ADIS.

    All public methods return safe defaults on any Redis or network
    error.  A cache failure is always treated as a cache miss — it
    never propagates to the API caller.
    """

    def __init__(self) -> None:
        # Created lazily on first property access (post-fork safe for Gunicorn)
        self._client: Optional[Redis] = None

    # ── Connection management ─────────────────────────────────────────────

    @property
    def client(self) -> Optional[Redis]:
        """
        Lazy-init the Redis connection pool on first access.

        ``Redis.from_url`` manages an internal thread-safe connection pool.
        Creating it post-fork ensures each Gunicorn worker owns its pool.

        Returns ``None`` (logged once) when Redis is unreachable.
        """
        if self._client is not None:
            return self._client

        try:
            self._client = self._build_client()
            self._client.ping()          # 1 command — validate at startup
            logger.info(
                "Redis client connected (url={})",
                self._redacted_url(),
            )
        except (RedisConnectionError, RedisTimeoutError, OSError) as exc:
            logger.warning(
                "Redis connection failed (url={}): {} — "
                "cache will be unavailable until Redis is reachable.",
                self._redacted_url(), exc,
            )
            self._client = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected Redis init error: {}", exc)
            self._client = None

        return self._client

    def _build_client(self) -> Redis:
        """
        Build a ``Redis`` instance from ``settings.REDIS_URL``.

        Handles ``redis://`` (local dev) and ``rediss://`` (Upstash TLS).
        When REDIS_SSL is True or the URL uses rediss://, relaxes cert
        verification for Upstash's managed certificates.
        """
        url = settings.REDIS_URL
        kwargs: Dict[str, Any] = {
            "decode_responses": True,               # str, not bytes
            "socket_timeout": settings.REDIS_SOCKET_TIMEOUT,
            "socket_connect_timeout": settings.REDIS_SOCKET_CONNECT_TIMEOUT,
            "socket_keepalive": True,               # survive Upstash idle timeouts
        }

        if settings.REDIS_SSL or url.startswith("rediss://"):
            # Upstash uses managed TLS certs — skip local CA verification
            kwargs["ssl_cert_reqs"] = "none"

        return Redis.from_url(url, **kwargs)

    def _is_available(self) -> bool:
        """Return True when the client is initialised and connected."""
        return self.client is not None

    def _reconnect(self) -> None:
        """
        Force reconnection on next ``client`` access.

        Called when a live operation raises a connection/timeout error,
        indicating the Redis connection was dropped (e.g. Upstash idle
        timeout after ~300 s of inactivity).
        """
        self._client = None

    def _redacted_url(self) -> str:
        """Return the Redis URL with any embedded password hidden."""
        url = settings.REDIS_URL
        if "@" in url:
            scheme, rest = url.split("://", 1)
            creds, host_part = rest.rsplit("@", 1)
            user = creds.split(":")[0] if ":" in creds else creds
            return f"{scheme}://{user}:***@{host_part}"
        return url

    # ── Serialisation helpers ─────────────────────────────────────────────

    @staticmethod
    def _serialise(data: Dict[str, Any]) -> str:
        """JSON-encode a dict to a UTF-8 string for Redis storage."""
        return json.dumps(data, default=str)

    @staticmethod
    def _deserialise(raw: str) -> Optional[Dict[str, Any]]:
        """Decode a Redis string back to a dict; None on malformed JSON."""
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Redis deserialise error: {}", exc)
            return None

    # ── SCAN + batch DEL helpers ──────────────────────────────────────────

    def _scan_keys(self, pattern: str) -> List[str]:
        """
        Collect all keys matching ``pattern`` via non-blocking SCAN
        iteration.  Safe on live production Redis — will not block the
        server like KEYS would.
        """
        if not self._is_available():
            return []

        collected: List[str] = []
        cursor: int = 0

        try:
            while True:
                cursor, batch = self.client.scan(  # type: ignore[union-attr]
                    cursor=cursor, match=pattern, count=_SCAN_COUNT
                )
                collected.extend(batch)
                if cursor == 0:
                    break
        except RedisError as exc:
            logger.warning("Redis SCAN error (pattern={}): {}", pattern, exc)

        return collected

    def _delete_keys(self, keys: List[str]) -> int:
        """
        Delete a list of keys in batches of ``_DEL_BATCH_SIZE``.

        Multi-key DEL is a single command per batch — minimises
        round-trips and conserves Upstash command budget.

        Returns total keys deleted.
        """
        if not keys or not self._is_available():
            return 0

        deleted = 0
        for i in range(0, len(keys), _DEL_BATCH_SIZE):
            batch = keys[i : i + _DEL_BATCH_SIZE]
            try:
                deleted += self.client.delete(*batch)  # type: ignore[union-attr]
            except RedisError as exc:
                logger.warning("Redis DEL batch error: {}", exc)

        return deleted

    def _silent_delete(self, key: str) -> None:
        """Delete a raw key, swallowing all errors (used for corrupt entries)."""
        try:
            self.client.delete(key)  # type: ignore[union-attr]
        except RedisError:
            pass

    # ══════════════════════════════════════════════════════════════════════
    # Analysis result cache — main hot path
    # ══════════════════════════════════════════════════════════════════════

    def get(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Return a cached analysis result for ``domain``.

        On a HIT, returns the stored dict with ``cached`` forced to
        ``True``.  On a MISS or any Redis error, returns ``None`` so
        the caller falls through to the full ML pipeline.

        Redis commands: 1 × GET
        """
        if not self._is_available():
            return None

        key = settings.redis_key(domain)

        try:
            raw = self.client.get(key)  # type: ignore[union-attr]
        except (RedisConnectionError, RedisTimeoutError) as exc:
            logger.warning(
                "Redis GET failed for '{}': {} — treating as cache miss",
                domain, exc,
            )
            self._reconnect()
            return None
        except RedisError as exc:
            logger.warning("Redis error on GET '{}': {}", key, exc)
            return None

        if raw is None:
            logger.debug("Cache MISS: '{}'", domain)
            return None

        result = self._deserialise(raw)
        if result is None:
            # Corrupt entry — purge and treat as miss
            logger.warning("Corrupt cache entry for '{}' — evicting", domain)
            self._silent_delete(key)
            return None

        result["cached"] = True
        logger.debug("Cache HIT: '{}' label='{}'", domain, result.get("label"))
        return result

    def set(
        self,
        domain: str,
        result: Dict[str, Any],
        label: str,
    ) -> bool:
        """
        Store an analysis result with a TTL determined by ``label``:

          'safe'        → 3,600 s  (settings.REDIS_TTL_SAFE)
          'suspicious'  →   900 s  (settings.REDIS_TTL_SUSPICIOUS)
          'malicious'   →   900 s  (settings.REDIS_TTL_MALICIOUS)

        The stored payload always has ``cached=False``.  ``cached=True``
        is injected by :meth:`get` at retrieval time.

        Redis commands: 1 × SET
        """
        if not self._is_available():
            return False

        key = settings.redis_key(domain)
        ttl = settings.cache_ttl_for_label(label)

        # Never store cached=True — only inject it on read
        payload = {**result, "cached": False}

        try:
            self.client.set(key, self._serialise(payload), ex=ttl)  # type: ignore[union-attr]
            logger.debug("Cache SET: '{}' label='{}' ttl={}s", domain, label, ttl)
            return True

        except (RedisConnectionError, RedisTimeoutError) as exc:
            logger.warning("Redis SET failed for '{}': {}", domain, exc)
            self._reconnect()
        except RedisError as exc:
            logger.warning("Redis error on SET '{}': {}", key, exc)

        return False

    def delete(self, domain: str) -> bool:
        """
        Remove the cached result for a single ``domain``.

        Returns ``True`` if the key existed and was deleted.

        Redis commands: 1 × DEL
        """
        if not self._is_available():
            return False

        key = settings.redis_key(domain)

        try:
            count = self.client.delete(key)  # type: ignore[union-attr]
            logger.debug("Cache DELETE: '{}' existed={}", domain, count > 0)
            return count > 0
        except RedisError as exc:
            logger.warning("Redis DEL error for '{}': {}", domain, exc)
            return False

    def exists(self, domain: str) -> bool:
        """
        Return ``True`` if a cached result exists for ``domain``.

        Calls Redis EXISTS (O(1)) — does not fetch the value.  Prefer
        this over ``get()`` when you only need a presence check.

        Redis commands: 1 × EXISTS
        """
        if not self._is_available():
            return False

        key = settings.redis_key(domain)

        try:
            return bool(self.client.exists(key))  # type: ignore[union-attr]
        except RedisError as exc:
            logger.warning("Redis EXISTS error for '{}': {}", domain, exc)
            return False

    def ttl_remaining(self, domain: str) -> int:
        """
        Return the remaining TTL (seconds) for a domain's cached result.

        Return values:
          ≥ 0   — seconds until expiry
          -1    — key exists with no expiry (unexpected in ADIS)
          -2    — key does not exist

        Redis commands: 1 × TTL
        """
        if not self._is_available():
            return -2

        key = settings.redis_key(domain)

        try:
            return self.client.ttl(key)  # type: ignore[union-attr]
        except RedisError as exc:
            logger.warning("Redis TTL error for '{}': {}", domain, exc)
            return -2

    # ══════════════════════════════════════════════════════════════════════
    # Network feature sub-caches  (FLAG 4 — WHOIS rate-limit mitigation)
    # ══════════════════════════════════════════════════════════════════════
    #
    # features/network.py calls get_whois_features() and get_dns_features()
    # BEFORE making any live network requests.  A cache HIT avoids the
    # 0.5–3s WHOIS round-trip entirely, preventing the ADIS server IP
    # from being rate-limited by registrar WHOIS servers (FLAG 4).
    #
    # WHOIS and DNS are stored in separate namespaces so each can be
    # invalidated independently without touching the other.
    #
    # Both use a 24-hour TTL as specified in FLAG 4.

    def get_whois_features(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Return cached WHOIS features for ``domain``, or ``None`` on miss.

        A HIT means features/network.py skips the live python-whois call
        entirely — protecting against WHOIS registrar rate bans (FLAG 4).

        Redis commands: 1 × GET
        """
        return self._get_sub_cache(CacheNamespace.WHOIS_RESULT, domain, "WHOIS")

    def set_whois_features(self, domain: str, features: Dict[str, Any]) -> bool:
        """
        Cache WHOIS feature data with a 24-hour TTL.

        Called by features/network.py after a successful python-whois
        lookup.  ``features`` should contain the computed WHOIS fields:
        domain_age_days, days_until_expiry, registration_length_days,
        registrar_is_common, whois_country, whois_privacy_enabled.

        Redis commands: 1 × SET
        """
        return self._set_sub_cache(
            CacheNamespace.WHOIS_RESULT, domain, features, _NETWORK_CACHE_TTL, "WHOIS"
        )

    def delete_whois_features(self, domain: str) -> bool:
        """Invalidate the WHOIS feature cache for a single domain."""
        return self._delete_sub_cache(CacheNamespace.WHOIS_RESULT, domain, "WHOIS")

    def get_dns_features(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Return cached DNS features for ``domain``, or ``None`` on miss.

        A HIT skips all dnspython queries for this domain — A, MX, NS,
        TXT, and AAAA records — saving ~50–200ms and reducing Upstash
        command consumption on repeat analyses.

        Redis commands: 1 × GET
        """
        return self._get_sub_cache(CacheNamespace.DNS_RESULT, domain, "DNS")

    def set_dns_features(self, domain: str, features: Dict[str, Any]) -> bool:
        """
        Cache DNS feature data with a 24-hour TTL.

        Called by features/network.py after successful DNS resolution.
        ``features`` should contain: has_a_record, num_a_records,
        has_mx_record, has_ns_record, num_ns_records, has_txt_record,
        dns_ttl, is_fast_flux, has_ipv6, dns_resolves.

        Redis commands: 1 × SET
        """
        return self._set_sub_cache(
            CacheNamespace.DNS_RESULT, domain, features, _NETWORK_CACHE_TTL, "DNS"
        )

    def delete_dns_features(self, domain: str) -> bool:
        """Invalidate the DNS feature cache for a single domain."""
        return self._delete_sub_cache(CacheNamespace.DNS_RESULT, domain, "DNS")

    # ── Shared sub-cache internals ────────────────────────────────────────

    def _get_sub_cache(
        self, namespace: str, domain: str, label: str
    ) -> Optional[Dict[str, Any]]:
        if not self._is_available():
            return None

        key = f"{namespace}{domain.lower()}"

        try:
            raw = self.client.get(key)  # type: ignore[union-attr]
        except (RedisConnectionError, RedisTimeoutError) as exc:
            logger.warning(
                "Redis GET failed for {} cache '{}': {} — miss",
                label, domain, exc,
            )
            self._reconnect()
            return None
        except RedisError as exc:
            logger.warning("Redis error on {} GET '{}': {}", label, key, exc)
            return None

        if raw is None:
            logger.debug("{} cache MISS: '{}'", label, domain)
            return None

        result = self._deserialise(raw)
        if result is None:
            logger.warning("Corrupt {} cache entry for '{}' — evicting", label, domain)
            self._silent_delete(key)
            return None

        logger.debug("{} cache HIT: '{}'", label, domain)
        return result

    def _set_sub_cache(
        self,
        namespace: str,
        domain: str,
        data: Dict[str, Any],
        ttl: int,
        label: str,
    ) -> bool:
        if not self._is_available():
            return False

        key = f"{namespace}{domain.lower()}"

        try:
            self.client.set(key, self._serialise(data), ex=ttl)  # type: ignore[union-attr]
            logger.debug("{} cache SET: '{}' ttl={}s", label, domain, ttl)
            return True
        except (RedisConnectionError, RedisTimeoutError) as exc:
            logger.warning("Redis SET failed for {} cache '{}': {}", label, domain, exc)
            self._reconnect()
        except RedisError as exc:
            logger.warning("Redis error on {} SET '{}': {}", label, key, exc)

        return False

    def _delete_sub_cache(self, namespace: str, domain: str, label: str) -> bool:
        if not self._is_available():
            return False

        key = f"{namespace}{domain.lower()}"

        try:
            count = self.client.delete(key)  # type: ignore[union-attr]
            logger.debug("{} cache DELETE: '{}' existed={}", label, domain, count > 0)
            return count > 0
        except RedisError as exc:
            logger.warning("Redis DEL error for {} cache '{}': {}", label, domain, exc)
            return False

    # ══════════════════════════════════════════════════════════════════════
    # Bulk operations  (POST /analyze/bulk — up to 50 domains)
    # ══════════════════════════════════════════════════════════════════════

    def get_multi(
        self, domains: List[str]
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Bulk-fetch cached results for multiple domains in one MGET call.

        Used by the POST /analyze/bulk endpoint to check all domains
        against cache before dispatching any ML inference.  A single MGET
        for 50 domains costs 1 command vs 50 individual GETs — critical
        for the Upstash 10K/day free-tier budget.

        Returns a dict mapping each domain to its cached result (or None).

        Redis commands: 1 × MGET
        """
        if not domains or not self._is_available():
            return {d: None for d in domains}

        keys = [settings.redis_key(d) for d in domains]

        try:
            raw_values = self.client.mget(keys)  # type: ignore[union-attr]
        except (RedisConnectionError, RedisTimeoutError) as exc:
            logger.warning("Redis MGET failed: {} — all misses", exc)
            self._reconnect()
            return {d: None for d in domains}
        except RedisError as exc:
            logger.warning("Redis MGET error: {}", exc)
            return {d: None for d in domains}

        results: Dict[str, Optional[Dict[str, Any]]] = {}
        hits = 0

        for domain, raw in zip(domains, raw_values):
            if raw is None:
                results[domain] = None
                continue
            parsed = self._deserialise(raw)
            if parsed is None:
                results[domain] = None
                continue
            parsed["cached"] = True
            results[domain] = parsed
            hits += 1

        logger.debug(
            "MGET: {}/{} hits for {} domains",
            hits, len(domains), len(domains),
        )
        return results

    def set_multi(
        self,
        entries: List[Tuple[str, Dict[str, Any], str]],
    ) -> int:
        """
        Bulk-write multiple analysis results via a Redis Pipeline.

        All SET commands are batched into a single round-trip.

        Each entry: ``(domain, result_dict, label)``

        Returns the number of entries successfully written.

        Redis commands: 1 × PIPELINE  (N SET commands inside, 1 round-trip)
        """
        if not entries or not self._is_available():
            return 0

        written = 0

        try:
            pipe = self.client.pipeline(transaction=False)  # type: ignore[union-attr]
            for domain, result, label in entries:
                key = settings.redis_key(domain)
                ttl = settings.cache_ttl_for_label(label)
                payload = {**result, "cached": False}
                pipe.set(key, self._serialise(payload), ex=ttl)

            responses = pipe.execute()
            written = sum(1 for r in responses if r)

        except (RedisConnectionError, RedisTimeoutError) as exc:
            logger.warning("Redis pipeline SET failed: {}", exc)
            self._reconnect()
        except RedisError as exc:
            logger.warning("Redis MULTI-SET error: {}", exc)

        logger.debug("Pipeline SET: {}/{} entries written", written, len(entries))
        return written

    # ══════════════════════════════════════════════════════════════════════
    # Admin operations  (POST /admin/cache/flush)
    # ══════════════════════════════════════════════════════════════════════

    def flush_domain_cache(self) -> int:
        """
        Delete all analysis result cache keys (adis:cache:*).

        Uses SCAN + batched DEL — never FLUSHDB — so rate-limit keys
        and network sub-caches are preserved.

        Called by POST /admin/cache/flush (typically after a model update
        to force all cached results to be re-computed).

        Returns the number of keys deleted.
        """
        pattern = f"{CacheNamespace.DOMAIN_RESULT}*"
        keys = self._scan_keys(pattern)
        deleted = self._delete_keys(keys)
        logger.info("flush_domain_cache: {} keys deleted", deleted)
        return deleted

    def flush_network_cache(self) -> int:
        """
        Delete all WHOIS and DNS feature cache keys.

        Call this to force fresh network lookups for all domains, e.g.
        after suspecting stale WHOIS registration data.

        Returns total keys deleted across both namespaces.
        """
        whois_keys = self._scan_keys(f"{CacheNamespace.WHOIS_RESULT}*")
        dns_keys   = self._scan_keys(f"{CacheNamespace.DNS_RESULT}*")
        all_keys   = whois_keys + dns_keys
        deleted    = self._delete_keys(all_keys)
        logger.info(
            "flush_network_cache: {} WHOIS + {} DNS = {} total deleted",
            len(whois_keys), len(dns_keys), deleted,
        )
        return deleted

    def flush_all(self) -> int:
        """
        Delete ALL adis:* keys — analysis results, WHOIS cache, DNS cache,
        and rate-limit keys.

        If you want to preserve flask-limiter rate-limit state, call
        flush_domain_cache() + flush_network_cache() instead.

        Returns total keys deleted.
        """
        keys = self._scan_keys("adis:*")
        deleted = self._delete_keys(keys)
        logger.info("flush_all: {} adis:* keys deleted", deleted)
        return deleted

    def flush_domain(self, domain: str) -> int:
        """
        Remove all cached data for a single ``domain``:
        the analysis result, WHOIS features, and DNS features.

        Returns the number of keys that existed and were deleted (0–3).

        Redis commands: 1 × DEL  (3 keys in one call)
        """
        if not self._is_available():
            return 0

        keys = [
            settings.redis_key(domain),
            f"{CacheNamespace.WHOIS_RESULT}{domain.lower()}",
            f"{CacheNamespace.DNS_RESULT}{domain.lower()}",
        ]

        try:
            deleted = self.client.delete(*keys)  # type: ignore[union-attr]
            logger.info("flush_domain '{}': {} keys deleted", domain, deleted)
            return deleted
        except RedisError as exc:
            logger.warning("Redis error flushing domain '{}': {}", domain, exc)
            return 0

    # ══════════════════════════════════════════════════════════════════════
    # Health check  (GET /health)
    # ══════════════════════════════════════════════════════════════════════

    def health_check(self) -> bool:
        """
        Verify Redis connectivity with a PING.

        On a failed PING, resets the client so the next call to
        ``client`` attempts a fresh reconnection.

        Returns ``True`` on success, ``False`` on any error.

        Redis commands: 1 × PING
        """
        # If not initialised, force a connection attempt now
        if not self._is_available():
            self._client = None   # ensure clean slate
            if not self._is_available():
                return False

        try:
            return bool(self.client.ping())  # type: ignore[union-attr]
        except (RedisConnectionError, RedisTimeoutError) as exc:
            logger.warning("Redis health check PING failed: {}", exc)
            self._reconnect()
            return False
        except RedisError as exc:
            logger.warning("Redis health check error: {}", exc)
            return False

    # ══════════════════════════════════════════════════════════════════════
    # Diagnostics  (GET /stats)
    # ══════════════════════════════════════════════════════════════════════

    def get_stats(self) -> CacheStats:
        """
        Return a :class:`CacheStats` snapshot of the cache state.

        Counts keys per namespace via SCAN, measures PING latency, and
        reads memory info from INFO.  Returns an empty CacheStats on
        any error — never raises.

        Note: key counts are approximate when many keys are expiring
        concurrently, which is normal during peak traffic.

        Redis commands: 1 × PING + 4 × SCAN passes + 2 × INFO
        """
        stats = CacheStats()

        if not self._is_available():
            return stats

        # ── PING latency ──────────────────────────────────────────────────
        try:
            t0 = time.perf_counter()
            self.client.ping()  # type: ignore[union-attr]
            stats.ping_ms = round((time.perf_counter() - t0) * 1000, 2)
            stats.connected = True
        except RedisError:
            # Can't even PING — return minimal stats
            return stats

        # ── Key counts per namespace ──────────────────────────────────────
        stats.domain_cache_keys = len(
            self._scan_keys(f"{CacheNamespace.DOMAIN_RESULT}*")
        )
        stats.whois_cache_keys = len(
            self._scan_keys(f"{CacheNamespace.WHOIS_RESULT}*")
        )
        stats.dns_cache_keys = len(
            self._scan_keys(f"{CacheNamespace.DNS_RESULT}*")
        )
        stats.total_adis_keys = len(self._scan_keys("adis:*"))

        # ── Memory info (non-critical — skip on error) ────────────────────
        try:
            mem_info = self.client.info("memory")  # type: ignore[union-attr]
            stats.used_memory_human = mem_info.get("used_memory_human", "unknown")
        except RedisError:
            pass

        # ── Redis version ─────────────────────────────────────────────────
        try:
            srv_info = self.client.info("server")  # type: ignore[union-attr]
            stats.redis_version = srv_info.get("redis_version", "unknown")
        except RedisError:
            pass

        return stats

    # ══════════════════════════════════════════════════════════════════════
    # Dunder
    # ══════════════════════════════════════════════════════════════════════

    def __repr__(self) -> str:
        return (
            f"<RedisClient url={self._redacted_url()!r} "
            f"connected={self._client is not None}>"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Singleton — import this everywhere
# ══════════════════════════════════════════════════════════════════════════════

cache = RedisClient()
