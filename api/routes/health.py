"""
api/routes/health.py
================================================================================
Public monitoring endpoints (blueprint section 10.1):

    GET /api/v1/health    Liveness/readiness check for uptime monitoring.
    GET /api/v1/version   Current model version info (+ registry metrics).

``/health`` is exempt from rate limiting so uptime monitors can poll freely.
It reports each subsystem (model / cache / database) and returns HTTP 200 when
the service can classify domains, or 503 when the model — the one hard
dependency — is not loaded. Cache and database are optimizations: if either is
down the service is "degraded" but still 200, because analysis still works
without them (the cache degrades to misses; logging is fire-and-forget).

``/version`` merges ``ModelServer.version_info()`` with the production row from
the Supabase ``model_versions`` registry when available. It carries no explicit
rate limit, so the limiter's default (RATE_LIMIT_DEFAULT, "100 per minute")
applies — matching section 10.3's "All others".

Real dependencies used (interfaces confirmed against the actual modules):
    api.extensions.limiter
    ml.model_server.get_model_server / ModelServer.is_loaded / .model_version
        / .version_info()
    cache.redis_client.cache.health_check() -> bool
    database.supabase_client.db.health_check() -> bool
    database.supabase_client.db.get_production_model() -> ModelVersionRow | None
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional, Tuple

from flask import Blueprint, jsonify
from loguru import logger

from api.extensions import limiter
from api.routes import utcnow_iso
from cache.redis_client import cache
from config.settings import settings
from database.supabase_client import db

health_bp = Blueprint("health", __name__, url_prefix=settings.API_PREFIX)


# =============================================================================
# Subsystem probes (never raise)
# =============================================================================
def _model_status() -> Tuple[bool, Optional[str], dict[str, Any]]:
    """
    Return ``(is_loaded, version, version_info)`` for the model singleton.

    ``get_model_server()`` constructs and eagerly loads the singleton on first
    call; if the model artefact is missing that raises, which we report as
    "down" rather than letting /health 500.
    """
    try:
        from ml.model_server import get_model_server

        server = get_model_server()
        return bool(server.is_loaded), server.model_version, server.version_info()
    except Exception as exc:
        logger.error("Health check: model server unavailable: {}", exc)
        return False, None, {}


def _cache_status() -> bool:
    try:
        return bool(cache.health_check())
    except Exception as exc:  # health_check never raises by design; belt & braces
        logger.warning("Health check: cache probe raised: {}", exc)
        return False


def _db_status() -> bool:
    try:
        return bool(db.health_check())
    except Exception as exc:
        logger.warning("Health check: database probe raised: {}", exc)
        return False


# =============================================================================
# Routes
# =============================================================================
@health_bp.get("/health")
@limiter.exempt
def health() -> Any:
    """Report subsystem health. 200 when serviceable, 503 when the model is down."""
    model_loaded, version, _ = _model_status()
    cache_ok = _cache_status()
    db_ok = _db_status()

    if model_loaded:
        status = "healthy" if (cache_ok and db_ok) else "degraded"
        http_status = 200
    else:
        status = "unhealthy"
        http_status = 503

    return (
        jsonify(
            {
                "status": status,
                "checks": {
                    "model": "up" if model_loaded else "down",
                    "cache": "up" if cache_ok else "down",
                    "database": "up" if db_ok else "down",
                },
                "model_version": version,
                "timestamp": utcnow_iso(),
            }
        ),
        http_status,
    )


@health_bp.get("/version")
def version() -> Any:
    """
    Return the running model's version info, enriched with the metrics of the
    production row in the ``model_versions`` registry when Supabase is up.
    """
    model_loaded, model_version, info = _model_status()

    payload: dict[str, Any] = {
        "model_version": model_version,
        "model_loaded": model_loaded,
        "feature_count": info.get("feature_count", settings.FEATURE_COUNT),
        "explainer_available": info.get("explainer_available", False),
        "timestamp": utcnow_iso(),
    }

    try:
        row = db.get_production_model()
        if row is not None:
            payload["metrics"] = {
                "accuracy": row.accuracy,
                "f1_score": row.f1_score,
                "auc_roc": row.auc_roc,
                "precision": row.precision_score,
                "recall": row.recall_score,
                "false_positive_rate": row.false_positive_rate,
                "training_samples": row.training_samples,
            }
            payload["deployed_at"] = row.deployed_at
            # The registry is authoritative for what *should* be in production;
            # surface a mismatch instead of silently preferring either side.
            payload["registry_version"] = row.version
            if model_version and row.version and row.version != model_version:
                payload["version_mismatch"] = True
                logger.warning(
                    "Loaded model version {} differs from registry production version {}",
                    model_version,
                    row.version,
                )
    except Exception as exc:  # db client shouldn't raise, but /version must not 500
        logger.warning("Could not load model registry row for /version: {}", exc)

    return jsonify(payload), 200
