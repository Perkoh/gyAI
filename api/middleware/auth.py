"""
api/middleware/auth.py
================================================================================
API-key authentication & authorization for the ADIS Flask API.

Per Section 10.1 of the ADIS blueprint, endpoints fall into three tiers:

    PUBLIC          (no key)     POST /analyze, GET /health, GET /version,
                                 POST /feedback
    AUTHENTICATED   (API key)    POST /analyze/bulk, GET /stats
    ADMIN           (admin key)  POST /admin/cache/flush,
                                 POST /admin/model/reload, GET /admin/logs

Authentication is done with the ``X-API-Key`` HTTP header. This module exposes
two decorators that guard the non-public tiers:

    @require_api_key      -> a valid API key OR admin key must be supplied
    @require_admin_key    -> a valid ADMIN key must be supplied

Usage
-----
    from api.middleware.auth import require_api_key, require_admin_key

    @bp.post("/analyze/bulk")
    @require_api_key
    def analyze_bulk():
        ...

    @bp.post("/admin/cache/flush")
    @require_admin_key
    def flush_cache():
        ...

On failure these decorators raise ``AuthenticationError`` (401) or
``AuthorizationError`` (403), which the global error handlers in
``error_handlers.py`` render into the standard ADIS JSON error envelope. The
decorators never build a response themselves — all error shaping is centralized.

Key configuration
-----------------
Valid keys are read from ``config.settings.settings`` at request time (so keys
can be rotated by restarting the process, and tests can monkeypatch settings):

    settings.API_KEYS         -> the authenticated-tier keys
    settings.ADMIN_API_KEYS   -> the admin-tier keys

Each may be provided as a list/tuple/set of strings, or as a single
comma-separated string (convenient for a ``.env`` value like
``ADIS_API_KEYS=key1,key2``). Admin keys are automatically also accepted on
authenticated endpoints — an admin can do anything an API-key holder can.

Security notes
--------------
* Keys are compared in constant time (``hmac.compare_digest``) to avoid leaking
  key material through timing side-channels.
* Endpoints fail *closed*: if no keys are configured on the server, every
  protected request is denied rather than silently allowed.
* Only a masked prefix of a rejected key is ever logged, never the full value.
"""

from __future__ import annotations

import hmac
from functools import wraps
from typing import Callable, Iterable, TypeVar

from flask import request
from loguru import logger

from api.middleware.error_handlers import AuthenticationError, AuthorizationError

# Importing the singleton settings object. Kept tolerant so the module is
# importable in isolated test setups; the getters below degrade to "no keys
# configured" (fail-closed) if settings or an attribute is absent.
try:
    from config.settings import settings
except Exception:  # pragma: no cover - settings always present in production
    settings = None  # type: ignore[assignment]


__all__ = ["require_api_key", "require_admin_key", "API_KEY_HEADER"]

#: The HTTP header carrying the caller's key (Section 10.1 of the blueprint).
API_KEY_HEADER = "X-API-Key"

F = TypeVar("F", bound=Callable[..., object])


# =============================================================================
# Key configuration helpers
# =============================================================================
def _coerce_key_set(raw: object) -> set[str]:
    """
    Normalize a configured key value into a set of non-empty strings.

    Accepts a comma-separated string, or any iterable of strings. ``None`` /
    empty / whitespace-only entries are dropped.
    """
    if not raw:
        return set()
    if isinstance(raw, str):
        candidates: Iterable[str] = raw.split(",")
    elif isinstance(raw, (list, tuple, set, frozenset)):
        candidates = [str(item) for item in raw]
    else:  # unexpected type — treat as no keys rather than crash
        logger.warning("Unexpected API key config type: {}", type(raw).__name__)
        return set()
    return {key.strip() for key in candidates if key and key.strip()}


def _get_admin_keys() -> set[str]:
    """The set of keys permitted on ADMIN-tier endpoints."""
    return _coerce_key_set(getattr(settings, "ADMIN_KEY", None))


def _get_api_keys() -> set[str]:
    """
    The set of keys permitted on AUTHENTICATED-tier endpoints.

    Admin keys are a superset: anyone holding an admin key may also call the
    ordinary authenticated endpoints.
    """
    return _coerce_key_set(getattr(settings, "API_KEY", None)) | _get_admin_keys()


# =============================================================================
# Key extraction / comparison
# =============================================================================
def _extract_api_key() -> str | None:
    """Pull the API key from the ``X-API-Key`` header, if present."""
    raw = request.headers.get(API_KEY_HEADER)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _key_matches(candidate: str, valid_keys: Iterable[str]) -> bool:
    """
    Constant-time membership test.

    Every configured key is checked (no early break) so that total comparison
    time does not depend on *which* key matched or how many precede it.
    """
    candidate_bytes = candidate.encode("utf-8")
    matched = False
    for key in valid_keys:
        if hmac.compare_digest(candidate_bytes, key.encode("utf-8")):
            matched = True
    return matched


def _mask(key: str) -> str:
    """Mask a key for safe logging, revealing at most a short prefix/suffix."""
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}…{key[-2:]}"


# =============================================================================
# Decorators
# =============================================================================
def require_api_key(func: F) -> F:
    """
    Guard an AUTHENTICATED-tier endpoint.

    Requires a valid ``X-API-Key`` matching ``settings.API_KEYS`` (or an admin
    key). Raises ``AuthenticationError`` (401) on a missing/invalid key.
    """

    @wraps(func)
    def wrapper(*args: object, **kwargs: object) -> object:
        key = _extract_api_key()
        if key is None:
            logger.warning(
                "Missing %s for authenticated endpoint %s", API_KEY_HEADER, request.path
            )
            raise AuthenticationError(
                f"A valid {API_KEY_HEADER} header is required to access this endpoint."
            )

        valid_keys = _get_api_keys()
        if not valid_keys:
            # Fail closed: nothing configured means nobody gets in.
            logger.error(
                "No API keys configured on the server; denying %s", request.path
            )
            raise AuthorizationError(
                "This endpoint is unavailable: no API credentials are configured on the server."
            )

        if not _key_matches(key, valid_keys):
            logger.warning(
                "Invalid API key %s for %s", _mask(key), request.path
            )
            raise AuthenticationError("The provided API key is invalid.")

        logger.debug("Authenticated request to %s", request.path)
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def require_admin_key(func: F) -> F:
    """
    Guard an ADMIN-tier endpoint.

    Requires a valid ``X-API-Key`` matching ``settings.ADMIN_API_KEYS``.

    Distinguishes three failure modes:
      * no key / unknown key           -> ``AuthenticationError`` (401)
      * a valid *non-admin* API key    -> ``AuthorizationError``  (403)
      * no admin keys configured       -> ``AuthorizationError``  (403, fail-closed)
    """

    @wraps(func)
    def wrapper(*args: object, **kwargs: object) -> object:
        key = _extract_api_key()
        if key is None:
            logger.warning(
                "Missing %s for admin endpoint %s", API_KEY_HEADER, request.path
            )
            raise AuthenticationError(
                f"A valid admin {API_KEY_HEADER} header is required to access this endpoint."
            )

        admin_keys = _get_admin_keys()
        if not admin_keys:
            logger.error(
                "No admin keys configured on the server; denying %s", request.path
            )
            raise AuthorizationError(
                "This endpoint is unavailable: no admin credentials are configured on the server."
            )

        if _key_matches(key, admin_keys):
            logger.info("Authorized admin request to %s", request.path)
            return func(*args, **kwargs)

        # A recognized but non-admin key is authenticated yet not privileged (403);
        # anything else is simply invalid (401).
        if _key_matches(key, _get_api_keys()):
            logger.warning(
                "Non-admin key %s attempted admin endpoint %s", _mask(key), request.path
            )
            raise AuthorizationError(
                "This endpoint requires administrator privileges."
            )

        logger.warning(
            "Invalid admin key %s for %s", _mask(key), request.path
        )
        raise AuthenticationError("The provided API key is invalid.")

    return wrapper  # type: ignore[return-value]
