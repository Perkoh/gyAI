"""
api/routes/analyze.py
================================================================================
The core gyAI analysis endpoints (blueprint sections 4.1, 4.2, 10.1–10.3):

    POST /api/v1/analyze        Public. One domain.        RATE_LIMIT_ANALYZE (per IP)
    POST /api/v1/analyze/bulk   API key. Up to 50 domains. RATE_LIMIT_BULK (per key)

Single-domain pipeline (mirrors the blueprint's request flow)::

    validate + normalize domain (Pydantic schema)
        -> cache.get(domain)          HIT -> return (client injects cached=True)
        -> db.lookup_known_domain     curated ground truth -> deterministic result
        -> assemble_feature_vector    48-dim; network features degrade to defaults
        -> ModelServer().predict      {score, label, confidence, reasons}
        -> db.log_analysis            fire-and-forget; skips safe internally
        -> cache.set(domain, ..., label)   label-driven TTL (1h safe / 15m flagged)

The bulk endpoint batches every stage: one ``cache.get_multi`` (1 MGET), one
``ModelServer().predict_batch`` model call, and one ``cache.set_multi``
(1 pipeline) — matching the Redis client's design for the Upstash free-tier
command budget. Per the ModelServer's design, bulk predictions skip SHAP reason
generation to keep latency low, so bulk results carry ``reasons: []``.

Notes on contract fields
------------------------
* ``analysis_id`` — ``db.log_analysis`` is fire-and-forget (returns ``None``),
  so the Supabase row id is unknowable at response time. For flagged domains
  the API instead returns a request-side UUID as the analysis identifier; safe
  domains get ``null`` (they are never logged, blueprint FLAG 1).
* ``network_features_used`` — the assembler returns a bare 48-element vector;
  feature #48 *is* ``network_features_available``, so it is read back out of
  the vector by index.
* ``label``/``confidence`` come verbatim from ``ModelServer.predict`` (the
  single source of truth on the ML path); the curated-list path derives its
  label via ``settings.score_to_label``.

Categorical encoders
--------------------
The assembler needs the encoders fitted at training time (``tld``,
``whois_country``). They are loaded once, best-effort, from
``settings.MODEL_DIR / "label_encoder.pkl"`` (the artefact named in blueprint
section 13). If absent, ``encoders=None`` triggers the assembler's documented
deterministic hash fallback and a warning is logged.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Mapping, Optional

from flask import Blueprint, jsonify
from loguru import logger
from pydantic import ValidationError

from api.extensions import (
    RATE_LIMIT_ANALYZE,
    RATE_LIMIT_BULK,
    api_key_or_ip,
    limiter,
)
from api.middleware.auth import require_api_key
from api.middleware.error_handlers import APIError, InvalidDomainError
from api.routes import first_error_message, require_json_object
from api.schemas.analyze_schema import (
    AnalyzeRequest,
    AnalyzeResponse,
    BulkAnalyzeRequest,
    BulkAnalyzeResponse,
)
from cache.redis_client import cache
from config.constants import LABEL, AnalysisSource, ErrorCode
from config.settings import settings
from database.supabase_client import db
from features.assembler import (
    FeatureExtractionError,
    assemble_feature_vector,
    feature_names,
)
from ml.model_server import ModelNotLoadedError, get_model_server

analyze_bp = Blueprint("analyze", __name__, url_prefix=settings.API_PREFIX)

# Index of the network_features_available meta-feature (feature #48) inside the
# assembled vector; resolved once from the canonical feature order.
_NET_AVAILABLE_IDX: int = feature_names().index("network_features_available")


# =============================================================================
# One-time, best-effort load of the fitted categorical encoders
# =============================================================================
def _load_encoders() -> Optional[Mapping[str, Any]]:
    path = settings.MODEL_DIR / "label_encoder.pkl"
    try:
        if not path.exists():
            logger.warning(
                "Categorical encoders not found at {}; the assembler will use "
                "its deterministic hash fallback. Ensure training used the same "
                "fallback, or ship label_encoder.pkl with the model.",
                path,
            )
            return None
        import joblib

        obj = joblib.load(path)
        if isinstance(obj, Mapping):
            logger.info("Loaded categorical encoders from {} ({})", path, list(obj))
            return obj
        logger.warning(
            "label_encoder.pkl did not contain a mapping of feature -> encoder "
            "(got {}); falling back to hash encoding.",
            type(obj).__name__,
        )
    except Exception as exc:
        logger.warning("Failed to load categorical encoders from {}: {}", path, exc)
    return None


_ENCODERS: Optional[Mapping[str, Any]] = _load_encoders()


# =============================================================================
# Pipeline pieces (shared by single + bulk)
# =============================================================================
def _response_from_cache_hit(domain: str, hit: Dict[str, Any], elapsed_ms: int) -> Optional[AnalyzeResponse]:
    """Validate a raw cache payload; return None (treat as miss) if malformed."""
    try:
        response = AnalyzeResponse.model_validate(hit)
    except Exception as exc:
        logger.warning("Discarding malformed cache entry for '{}': {}", domain, exc)
        return None
    # The Redis client injects cached=True on read; just restamp the duration.
    return response.model_copy(update={"duration_ms": max(elapsed_ms, 0)})


def _response_from_known_domain(domain: str, verdict: str, source: Optional[str]) -> AnalyzeResponse:
    """
    Deterministic result for a curated ``known_domains`` entry (the short-
    circuit the Supabase client documents for this route). 'safe' maps to score
    0.0; 'malicious'/'phishing' map to 1.0.
    """
    is_bad = verdict != "safe"
    score = 1.0 if is_bad else 0.0
    label = settings.score_to_label(score)
    reasons = (
        [f"This domain is on a curated threat list (source: {source or 'manual'})"]
        if is_bad
        else []
    )
    return AnalyzeResponse(
        domain=domain,
        score=score,
        label=label,
        confidence="high",
        reasons=reasons,
        model_version=_model_version_safe(),
        analysis_id=str(uuid.uuid4()) if is_bad else None,
        duration_ms=0,
        cached=False,
        network_features_used=False,
    )


def _model_version_safe() -> str:
    try:
        return get_model_server().model_version
    except Exception:
        return settings.MODEL_VERSION


def _assemble_vector(domain: str):
    """Assemble the 48-dim vector, translating failures into API errors."""
    try:
        return assemble_feature_vector(domain, encoders=_ENCODERS)
    except FeatureExtractionError as exc:
        # The schema already validated the domain, so this indicates an
        # extractor problem rather than bad user input.
        logger.error("Feature assembly failed for '{}': {}", domain, exc)
        raise APIError(
            "Analysis failed while extracting features for this domain.",
            code=ErrorCode.ANALYSIS_FAILED,
            status_code=500,
        )


def _predict_single(vector) -> Dict[str, Any]:
    try:
        return get_model_server().predict(vector, explain=True, top_reasons=settings.SHAP_MAX_REASONS)
    except ModelNotLoadedError:
        raise APIError(
            "The analysis model is not loaded. Please try again shortly.",
            code=ErrorCode.MODEL_NOT_LOADED,
            status_code=503,
        )


def _build_response(
    domain: str,
    prediction: Dict[str, Any],
    *,
    network_used: bool,
    duration_ms: int,
) -> AnalyzeResponse:
    label = prediction["label"]
    flagged = label != LABEL.SAFE
    return AnalyzeResponse(
        domain=domain,
        score=prediction["score"],
        label=label,
        confidence=prediction["confidence"],
        reasons=prediction.get("reasons") or [],
        model_version=_model_version_safe(),
        analysis_id=str(uuid.uuid4()) if flagged else None,
        duration_ms=max(duration_ms, 0),
        cached=False,
        network_features_used=network_used,
    )


def _log_flagged(response: AnalyzeResponse, source: AnalysisSource) -> None:
    """Fire-and-forget Supabase log. The client itself skips safe domains."""
    try:
        db.log_analysis(
            domain=response.domain,
            score=response.score,
            label=response.label,
            confidence=response.confidence,
            reasons=response.reasons,
            model_version=response.model_version,
            duration_ms=response.duration_ms,
            network_features_used=response.network_features_used,
            source=source,
        )
    except Exception as exc:  # the client is fault-tolerant; belt & braces
        logger.warning("Supabase log dispatch failed for '{}': {}", response.domain, exc)


def _analyze_domain(domain: str, *, source: AnalysisSource) -> AnalyzeResponse:
    """Full single-domain pipeline: cache -> known_domains -> features -> model."""
    start = time.perf_counter()

    # 1. Redis cache (client returns cached=True-injected payload on hit).
    hit = cache.get(domain)
    if hit is not None:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        response = _response_from_cache_hit(domain, hit, elapsed_ms)
        if response is not None:
            return response

    # 2. Curated ground truth short-circuits ML inference.
    known = db.lookup_known_domain(domain)
    if known is not None:
        response = _response_from_known_domain(domain, known.verdict, known.source)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        response = response.model_copy(update={"duration_ms": elapsed_ms})
        _log_flagged(response, source)
        cache.set(domain, response.model_dump(mode="json"), response.label)
        return response

    # 3. Feature extraction + model inference.
    vector = _assemble_vector(domain)
    network_used = bool(vector[_NET_AVAILABLE_IDX] >= 0.5)
    prediction = _predict_single(vector)

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    response = _build_response(
        domain, prediction, network_used=network_used, duration_ms=elapsed_ms
    )

    # 4. Log (skips safe internally, async) + cache with label-driven TTL.
    _log_flagged(response, source)
    cache.set(domain, response.model_dump(mode="json"), response.label)
    return response


# =============================================================================
# Routes
# =============================================================================
@analyze_bp.post("/analyze")
@limiter.limit(RATE_LIMIT_ANALYZE)
def analyze() -> Any:
    """Analyze a single domain (public; the browser extension's hot path)."""
    data = require_json_object()
    try:
        req = AnalyzeRequest.model_validate(data)
    except ValidationError as exc:
        raise InvalidDomainError(first_error_message(exc))

    result = _analyze_domain(req.domain, source=AnalysisSource.EXTENSION)
    return jsonify(result.model_dump(mode="json")), 200


@analyze_bp.post("/analyze/bulk")
@limiter.limit(RATE_LIMIT_BULK, key_func=api_key_or_ip)
@require_api_key
def analyze_bulk() -> Any:
    """
    Analyze up to ``settings.BULK_MAX_DOMAINS`` domains (requires an API key).

    Batched I/O: 1 MGET for cache reads, 1 model call, 1 pipeline for cache
    writes. Reasons are omitted in bulk (ModelServer.predict_batch default).
    """
    data = require_json_object()
    try:
        req = BulkAnalyzeRequest.model_validate(data)
    except ValidationError as exc:
        raise InvalidDomainError(first_error_message(exc))

    start = time.perf_counter()
    results_by_domain: Dict[str, AnalyzeResponse] = {}

    # ── Stage 1: one MGET for all domains ─────────────────────────────────
    cached = cache.get_multi(req.domains)
    misses: List[str] = []
    for domain in req.domains:
        hit = cached.get(domain)
        response = _response_from_cache_hit(domain, hit, 0) if hit else None
        if response is not None:
            results_by_domain[domain] = response
        else:
            misses.append(domain)

    # ── Stage 2: curated list, features, and one batched model call ───────
    to_predict: List[str] = []
    vectors: List[Any] = []
    network_flags: Dict[str, bool] = {}

    for domain in misses:
        known = db.lookup_known_domain(domain)
        if known is not None:
            results_by_domain[domain] = _response_from_known_domain(
                domain, known.verdict, known.source
            )
            continue
        vector = _assemble_vector(domain)
        network_flags[domain] = bool(vector[_NET_AVAILABLE_IDX] >= 0.5)
        to_predict.append(domain)
        vectors.append(vector)

    if vectors:
        try:
            predictions = get_model_server().predict_batch(vectors)
        except ModelNotLoadedError:
            raise APIError(
                "The analysis model is not loaded. Please try again shortly.",
                code=ErrorCode.MODEL_NOT_LOADED,
                status_code=503,
            )
        compute_ms = int((time.perf_counter() - start) * 1000)
        per_domain_ms = compute_ms // max(len(to_predict), 1)  # approximation
        for domain, prediction in zip(to_predict, predictions):
            results_by_domain[domain] = _build_response(
                domain,
                prediction,
                network_used=network_flags[domain],
                duration_ms=per_domain_ms,
            )

    # ── Stage 3: log flagged + one pipelined cache write ──────────────────
    fresh = [results_by_domain[d] for d in req.domains if not results_by_domain[d].cached]
    for response in fresh:
        _log_flagged(response, AnalysisSource.API)
    if fresh:
        cache.set_multi(
            [(r.domain, r.model_dump(mode="json"), r.label) for r in fresh]
        )

    total_ms = int((time.perf_counter() - start) * 1000)
    ordered = [results_by_domain[d] for d in req.domains]
    body = BulkAnalyzeResponse(count=len(ordered), duration_ms=max(total_ms, 0), results=ordered)
    logger.info(
        "Bulk analyzed {} domain(s) ({} cache hits) in {}ms",
        len(ordered),
        len(ordered) - len(misses),
        total_ms,
    )
    return jsonify(body.model_dump(mode="json")), 200
