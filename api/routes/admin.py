"""
api/routes/admin.py
================================================================================
Administrative and statistics endpoints (blueprint section 10.1):

    GET  /api/v1/stats              Aggregated system stats.    (API key)
    POST /api/v1/admin/cache/flush  Clear the Redis cache.      (admin key)
    POST /api/v1/admin/model/reload Reload the model from disk. (admin key)
    GET  /api/v1/admin/logs         Recent flagged-domain log.  (admin key)

``/stats`` is an *authenticated*-tier endpoint (any valid API key), not an
admin one — the blueprint's file layout has no ``stats.py``, so it lives here
alongside the admin routes, guarded by ``require_api_key`` while ``/admin/*``
uses ``require_admin_key``.

None of these routes declares an explicit rate limit: the limiter's default
(RATE_LIMIT_DEFAULT, "100 per minute") applies, matching section 10.3's
"All others".

Cache flush scopes
------------------
The Redis client exposes several flush granularities; the endpoint accepts an
optional JSON body selecting one (default ``results``)::

    {"scope": "results"}                  -> cache.flush_domain_cache()
    {"scope": "network"}                  -> cache.flush_network_cache()
    {"scope": "all"}                      -> cache.flush_all()   (includes
                                             rate-limit keys — use sparingly)
    {"scope": "domain", "domain": "x.y"}  -> cache.flush_domain("x.y")

Real dependencies used (interfaces confirmed against the actual modules):
    cache.redis_client.cache.flush_domain_cache/-network_cache/-_all/-_domain,
        .get_stats() -> CacheStats
    database.supabase_client.db.get_recent_analyses(limit, label)
        -> list[DomainAnalysisRow], .get_stats() -> AnalysisStats
    ml.model_server.get_model_server().reload() -> version_info dict
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from flask import Blueprint, jsonify, request
from loguru import logger

from api.middleware.auth import require_admin_key, require_api_key
from api.middleware.error_handlers import BadRequestError, InternalServerError
from api.routes import int_arg, utcnow_iso
from cache.redis_client import cache
from config.constants import LABEL
from config.settings import settings
from database.supabase_client import db

admin_bp = Blueprint("admin", __name__, url_prefix=settings.API_PREFIX)

#: Labels that appear in domain_analyses (safe domains are never logged).
_LOGGABLE_LABELS = {LABEL.SUSPICIOUS, LABEL.MALICIOUS}

_FLUSH_SCOPES = {"results", "network", "all", "domain"}


# =============================================================================
# Statistics (authenticated tier)
# =============================================================================
@admin_bp.get("/stats")
@require_api_key
def stats() -> Any:
    """
    Aggregated system statistics: Supabase analysis aggregates plus a Redis
    cache snapshot. Both clients degrade to empty stats objects on outage, so
    this endpoint reports what it can rather than failing.
    """
    try:
        analysis_stats = asdict(db.get_stats())
    except Exception as exc:
        logger.warning("Failed to load analysis stats: {}", exc)
        analysis_stats = {}

    try:
        cache_stats = asdict(cache.get_stats())
    except Exception as exc:
        logger.warning("Failed to load cache stats: {}", exc)
        cache_stats = {}

    return (
        jsonify(
            {
                "analysis": analysis_stats,
                "cache": cache_stats,
                "model_version": settings.MODEL_VERSION,
                "timestamp": utcnow_iso(),
            }
        ),
        200,
    )


# =============================================================================
# Admin tier
# =============================================================================
@admin_bp.post("/admin/cache/flush")
@require_admin_key
def flush_cache() -> Any:
    """Clear Redis cache entries (scope selectable via optional JSON body)."""
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        raise BadRequestError("Request body, when provided, must be a JSON object.")

    scope = str(body.get("scope", "results")).strip().lower()
    if scope not in _FLUSH_SCOPES:
        raise BadRequestError(
            f"Invalid scope '{scope}'. Must be one of: {', '.join(sorted(_FLUSH_SCOPES))}."
        )

    domain: Optional[str] = None
    try:
        if scope == "results":
            removed = cache.flush_domain_cache()
        elif scope == "network":
            removed = cache.flush_network_cache()
        elif scope == "all":
            removed = cache.flush_all()
        else:  # scope == "domain"
            domain = str(body.get("domain", "")).strip().lower()
            if not domain:
                raise BadRequestError("Scope 'domain' requires a 'domain' field.")
            removed = cache.flush_domain(domain)
    except BadRequestError:
        raise
    except Exception as exc:
        logger.error("Cache flush ({}) failed: {}", scope, exc)
        raise InternalServerError("Failed to flush the cache.")

    logger.info("Admin cache flush scope='{}' domain={} removed={}", scope, domain, removed)
    return (
        jsonify(
            {
                "status": "ok",
                "scope": scope,
                "domain": domain,
                "keys_removed": removed,
                "timestamp": utcnow_iso(),
            }
        ),
        200,
    )


@admin_bp.post("/admin/model/reload")
@require_admin_key
def reload_model() -> Any:
    """
    Reload the LightGBM model from disk without restarting the process.
    ``ModelServer.reload()`` swaps the model atomically and returns the fresh
    ``version_info()`` payload, which is echoed back.
    """
    try:
        from ml.model_server import get_model_server

        info = get_model_server().reload()
    except FileNotFoundError as exc:
        logger.error("Model reload failed — artefact missing: {}", exc)
        raise InternalServerError("Model file not found on disk; reload aborted.")
    except Exception as exc:
        logger.error("Model reload failed: {}", exc)
        raise InternalServerError("Failed to reload the model from disk.")

    logger.info("Admin reloaded model: {}", info)
    return (
        jsonify({"status": "reloaded", "model": info, "timestamp": utcnow_iso()}),
        200,
    )


@admin_bp.get("/admin/logs")
@require_admin_key
def recent_logs() -> Any:
    """
    Most recent flagged-domain analyses, newest first.

    Query parameters:
        limit  int, 1–500 (default 50; the Supabase client caps at 500)
        label  optional filter: 'suspicious' | 'malicious'
    """
    limit = int_arg("limit", default=50, min_=1, max_=500)

    label = request.args.get("label")
    if label is not None:
        label = label.strip().lower()
        if label not in _LOGGABLE_LABELS:
            raise BadRequestError(
                "Query parameter 'label' must be 'suspicious' or 'malicious'."
            )

    try:
        rows = db.get_recent_analyses(limit=limit, label=label)
    except Exception as exc:
        logger.error("Failed to fetch recent analyses: {}", exc)
        raise InternalServerError("Unable to retrieve logs at this time.")

    logs = [asdict(row) for row in rows]
    return (
        jsonify(
            {
                "count": len(logs),
                "limit": limit,
                "label": label,
                "logs": logs,
                "timestamp": utcnow_iso(),
            }
        ),
        200,
    )
