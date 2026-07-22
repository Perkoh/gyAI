"""
gyAI — AI-Powered Domain Intelligence System
api/extensions.py

Initialises the shared Flask extensions used across the API (blueprint
section 13): CORS, rate limiting, and the Redis client.

These objects are created at import time so route modules can import and use
them directly, e.g.:

    from api.extensions import limiter, RATE_LIMIT_ANALYZE

    @analyze_bp.post("/analyze")
    @limiter.limit(RATE_LIMIT_ANALYZE)
    def analyze(): ...

`init_extensions(app)` is called once from the application factory
(api/app.py::create_app) to bind everything to the app using values from
app.config.

Key blueprint requirements honoured here:
  * FLAG 6 — CORS origins are restricted to browser-extension schemes
    (chrome-extension://*, moz-extension://*), never a bare "*" in production.
  * FLAG 3 / section 10.3 — Rate limits are stored in Redis so they are shared
    across all Gunicorn workers (an in-memory store would let each of the 4
    workers grant the full quota independently).
"""

from __future__ import annotations

from typing import List, Optional

import redis
from flask import Flask, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

try:
    from loguru import logger
except Exception:  # pragma: no cover - loguru is a hard dep; degrade gracefully
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("adis.extensions")  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Rate-limit strings (blueprint section 10.3). Imported by the route modules
# so the numbers live in exactly one place.
# --------------------------------------------------------------------------- #
RATE_LIMIT_DEFAULT = "100 per minute"    # "All others"
RATE_LIMIT_ANALYZE = "60 per minute"     # POST /analyze  (per IP)
RATE_LIMIT_BULK = "10 per minute"        # POST /analyze/bulk (per API key)
RATE_LIMIT_FEEDBACK = "20 per minute"    # POST /feedback (per IP)

# Default CORS origins: browser extensions only. Passed to flask-cors as regex
# patterns (flask-cors matches each origin string as a regex).
DEFAULT_CORS_ORIGINS: List[str] = [
    r"chrome-extension://.*",
    r"moz-extension://.*",
]

# Redis cache key prefix (blueprint section 4.2: "adis:cache:<domain>").
CACHE_KEY_PREFIX = "adis:cache:"


# --------------------------------------------------------------------------- #
# Extension singletons
# --------------------------------------------------------------------------- #
cors = CORS()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[RATE_LIMIT_DEFAULT],
    headers_enabled=True,
)


class _RedisHolder:
    """Holds the process-wide Redis client once initialised."""
    client: Optional["redis.Redis"] = None


_redis = _RedisHolder()


def api_key_or_ip() -> str:
    """
    Rate-limit key function for endpoints limited *per API key* (e.g. bulk).
    Falls back to client IP when no key is supplied.
    """
    key = request.headers.get("X-API-Key")
    return f"key:{key}" if key else f"ip:{get_remote_address()}"


# --------------------------------------------------------------------------- #
# Init entry point
# --------------------------------------------------------------------------- #
def init_extensions(app: Flask) -> None:
    """Bind CORS, the rate limiter, and Redis to the application."""
    _init_cors(app)
    _init_limiter(app)
    _init_redis(app)


def _init_cors(app: Flask) -> None:
    origins = app.config.get("CORS_ORIGINS") or list(DEFAULT_CORS_ORIGINS)
    extra = app.config.get("CORS_EXTRA_ORIGINS")
    if extra:
        origins = list(origins) + list(extra)

    cors.init_app(
        app,
        resources={r"/api/*": {"origins": origins}},
        methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key"],
        supports_credentials=False,
    )
    logger.info(f"CORS enabled for /api/* with origins: {origins}")


def _init_limiter(app: Flask) -> None:
    # Share the limiter store across Gunicorn workers via Redis. Without a
    # shared store each worker enforces the quota independently.
    storage_uri = (
        app.config.get("RATELIMIT_STORAGE_URI")
        or app.config.get("REDIS_URL")
        or "memory://"
    )
    app.config.setdefault("RATELIMIT_STORAGE_URI", storage_uri)
    app.config.setdefault("RATELIMIT_DEFAULT", RATE_LIMIT_DEFAULT)
    app.config.setdefault("RATELIMIT_HEADERS_ENABLED", True)
    # Fail open: if the limiter's backing store is unreachable, allow the
    # request through rather than 500-ing a security tool the user relies on.
    app.config.setdefault("RATELIMIT_IN_MEMORY_FALLBACK_ENABLED", True)
    app.config.setdefault("RATELIMIT_SWALLOW_ERRORS", True)

    if storage_uri.startswith("memory://"):
        logger.warning(
            "Rate limiter is using an in-memory store — limits will NOT be "
            "shared across Gunicorn workers. Set REDIS_URL for production."
        )

    limiter.init_app(app)
    logger.info(f"Rate limiter initialised (storage={storage_uri}).")


def _init_redis(app: Flask) -> None:
    url = app.config.get("REDIS_URL")
    if not url:
        logger.warning("REDIS_URL not configured; Redis cache is disabled.")
        _redis.client = None
        return

    timeout = float(app.config.get("REDIS_SOCKET_TIMEOUT", 2.0))
    try:
        client = redis.from_url(
            url,
            decode_responses=True,
            socket_timeout=timeout,
            socket_connect_timeout=timeout,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        # Verify connectivity, but don't crash the app if Redis is down at
        # boot — /health will report the degraded state and the analyze route
        # can fall back to computing without the cache.
        try:
            client.ping()
            logger.info("Redis connection established.")
        except Exception as exc:
            logger.warning(f"Redis ping failed at startup ({exc}); caching degraded.")
        _redis.client = client
    except Exception as exc:
        logger.error(f"Failed to initialise Redis client: {exc}")
        _redis.client = None

    app.extensions["redis"] = _redis.client


# --------------------------------------------------------------------------- #
# Accessors / health
# --------------------------------------------------------------------------- #
def get_redis() -> Optional["redis.Redis"]:
    """Return the shared Redis client, or None if unavailable/unconfigured."""
    return _redis.client


def redis_healthy() -> bool:
    """True if Redis is configured and currently responding to PING."""
    client = _redis.client
    if client is None:
        return False
    try:
        return bool(client.ping())
    except Exception:
        return False