"""
api/routes
================================================================================
The ADIS HTTP route layer. Each endpoint group is a Flask ``Blueprint``:

    api/routes/health.py    ->  health_bp    (GET  /health, GET /version)
    api/routes/analyze.py   ->  analyze_bp   (POST /analyze, POST /analyze/bulk)
    api/routes/feedback.py  ->  feedback_bp  (POST /feedback)
    api/routes/admin.py     ->  admin_bp     (GET /stats, POST /admin/*, GET /admin/logs)

All blueprints mount under ``settings.API_PREFIX`` (``/api/v1``). The app
factory registers them::

    from api.routes.health import health_bp
    from api.routes.analyze import analyze_bp
    from api.routes.feedback import feedback_bp
    from api.routes.admin import admin_bp

    for bp in (health_bp, analyze_bp, feedback_bp, admin_bp):
        app.register_blueprint(bp)

This ``__init__`` holds only small helpers shared across the blueprints. It
deliberately does not import the route modules (they import from here, so that
would be a cycle).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from flask import request

from api.middleware.error_handlers import BadRequestError

__all__ = ["utcnow_iso", "require_json_object", "first_error_message", "int_arg"]


def utcnow_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def require_json_object() -> dict[str, Any]:
    """
    Parse the request body as a JSON object, or raise :class:`BadRequestError`.

    Guards against a missing body, a non-JSON body, and a JSON value that is
    not an object (e.g. a bare list or string), before the data reaches the
    Pydantic schema.
    """
    data = request.get_json(silent=True)
    if data is None:
        raise BadRequestError("Request body must be valid JSON.")
    if not isinstance(data, dict):
        raise BadRequestError("Request body must be a JSON object.")
    return data


def first_error_message(exc: Any) -> str:
    """
    Extract a clean human-readable message from a Pydantic ``ValidationError``.

    Pydantic prefixes messages raised inside custom validators with
    ``"Value error, "``; that prefix is stripped so the surfaced message reads
    naturally (e.g. ``"domain must contain at least one dot"``).
    """
    fallback = "The provided value is not a valid domain name."
    try:
        errors = exc.errors()
    except Exception:
        return fallback
    if not errors:
        return fallback
    msg = str(errors[0].get("msg", fallback))
    for prefix in ("Value error, ", "Assertion failed, "):
        if msg.startswith(prefix):
            msg = msg[len(prefix):]
    return msg or fallback


def int_arg(name: str, *, default: int, min_: int, max_: int) -> int:
    """
    Read an integer query-string argument, clamped to ``[min_, max_]``.

    Raises :class:`BadRequestError` if the value is present but not an integer.
    """
    raw = request.args.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise BadRequestError(f"Query parameter '{name}' must be an integer.")
    return max(min_, min(max_, value))
