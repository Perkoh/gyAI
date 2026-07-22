"""
api/middleware/error_handlers.py
================================================================================
Global error handling for the ADIS Flask API.

Every error the API returns — whether raised deliberately by our own code, by
Pydantic validation, by Flask-Limiter rate limiting, or by an unexpected crash —
is funnelled through here and rendered as a single, consistent JSON envelope:

    {
        "error": {
            "code": "INVALID_DOMAIN",
            "message": "The provided value is not a valid domain name",
            "status": 422
        }
    }

This matches the contract defined in Section 10.2 of the ADIS blueprint.

Two things live in this module:

1.  An `APIError` exception hierarchy. Application code (routes, middleware,
    services) raises these instead of calling `flask.abort()` or returning ad-hoc
    error dicts. Each carries a machine-readable `code`, a human-readable
    `message`, and an HTTP `status`.

2.  `register_error_handlers(app)` — call this once from the application factory
    (`api/app.py :: create_app`) to wire every handler onto the Flask app.

Design notes
------------
* 4xx errors are treated as *client* problems: their messages are safe to return
  verbatim, and they are logged at WARNING level (or below).
* 5xx errors are treated as *server* problems: the real exception is logged with
  a full traceback, but the client only ever sees a generic message so we never
  leak internals (stack traces, DB errors, secrets) across the wire.
* The catch-all `Exception` handler guarantees the API *always* answers with the
  JSON envelope, never an HTML Werkzeug error page — important because the only
  consumers are the browser extension and programmatic API clients.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from flask import Flask, jsonify, request
from flask.wrappers import Response
from loguru import logger
from werkzeug.exceptions import HTTPException

# Pydantic is a hard dependency of the project (see requirements.txt). The import
# is guarded only so this module can be imported in stripped-down environments
# (e.g. isolated unit tests) without pulling in Pydantic.
try:
    from pydantic import ValidationError as PydanticValidationError
except Exception:  # pragma: no cover - pydantic always present in production
    PydanticValidationError = None  # type: ignore[assignment, misc]

# Flask-Limiter raises RateLimitExceeded (a 429 HTTPException subclass) when a
# limit is tripped. Guarded for the same reason as above.
try:
    from flask_limiter.errors import RateLimitExceeded
except Exception:  # pragma: no cover - flask-limiter always present in production
    RateLimitExceeded = None  # type: ignore[assignment, misc]

if TYPE_CHECKING:
    from pydantic import ValidationError as PydanticValidationError  # type: ignore[assignment, misc]
    from flask_limiter.errors import RateLimitExceeded  # type: ignore[assignment, misc]


__all__ = [
    "APIError",
    "BadRequestError",
    "InvalidDomainError",
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "PayloadTooLargeError",
    "RateLimitError",
    "InternalServerError",
    "register_error_handlers",
]


# =============================================================================
# Exception hierarchy
# =============================================================================
class APIError(Exception):
    """
    Base class for every error the ADIS API raises on purpose.

    Subclasses set sensible defaults for ``code``, ``message`` and
    ``status_code``, but any of them can be overridden at raise-time::

        raise InvalidDomainError("'%s' is not a domain" % value)
        raise APIError("Teapot", code="IM_A_TEAPOT", status_code=418)

    ``details`` is an optional list/dict of extra structured context (used, for
    example, to attach per-field Pydantic validation errors). It is included in
    the response only when present.
    """

    #: Default machine-readable error code (UPPER_SNAKE_CASE).
    code: str = "API_ERROR"
    #: Default human-readable message. Safe to show to end users.
    message: str = "An error occurred while processing the request."
    #: Default HTTP status code.
    status_code: int = 400

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: Any = None,
    ) -> None:
        self.message = message if message is not None else self.message
        self.code = code if code is not None else self.code
        self.status_code = status_code if status_code is not None else self.status_code
        self.details = details
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Render this error into the ADIS JSON error envelope."""
        error: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "status": self.status_code,
        }
        if self.details is not None:
            error["details"] = self.details
        return {"error": error}


class BadRequestError(APIError):
    """400 — the request was malformed (bad JSON, missing body, wrong type)."""

    code = "BAD_REQUEST"
    message = "The request could not be understood or was missing required data."
    status_code = 400


class InvalidDomainError(APIError):
    """422 — the payload parsed but a value failed validation (e.g. bad domain)."""

    code = "INVALID_DOMAIN"
    message = "The provided value is not a valid domain name."
    status_code = 422


class AuthenticationError(APIError):
    """401 — the API key is missing or invalid."""

    code = "UNAUTHORIZED"
    message = "A valid API key is required to access this endpoint."
    status_code = 401


class AuthorizationError(APIError):
    """403 — the caller is authenticated but lacks the required privilege."""

    code = "FORBIDDEN"
    message = "You do not have permission to access this endpoint."
    status_code = 403


class NotFoundError(APIError):
    """404 — the requested resource or route does not exist."""

    code = "NOT_FOUND"
    message = "The requested resource was not found."
    status_code = 404


class PayloadTooLargeError(APIError):
    """413 — the request exceeds a size limit (e.g. > 50 domains in /analyze/bulk)."""

    code = "PAYLOAD_TOO_LARGE"
    message = "The request payload exceeds the maximum allowed size."
    status_code = 413


class RateLimitError(APIError):
    """429 — the caller has exceeded the configured rate limit."""

    code = "RATE_LIMIT_EXCEEDED"
    message = "Rate limit exceeded. Please slow down and try again shortly."
    status_code = 429


class InternalServerError(APIError):
    """500 — an unexpected server-side failure."""

    code = "INTERNAL_ERROR"
    message = "An internal error occurred. Please try again later."
    status_code = 500


# =============================================================================
# Helpers
# =============================================================================
def _make_response(payload: dict[str, Any], status: int) -> Response:
    """Build a JSON Flask ``Response`` with the given status code."""
    response = jsonify(payload)
    response.status_code = status
    return response


def _client_context() -> str:
    """A short, log-friendly description of the current request."""
    try:
        return f"{request.method} {request.path} from {request.remote_addr}"
    except Exception:  # pragma: no cover - outside a request context
        return "<no request context>"


# =============================================================================
# Registration
# =============================================================================
def register_error_handlers(app: Flask) -> None:
    """
    Attach all ADIS error handlers to a Flask application.

    Call once from the application factory::

        from api.middleware.error_handlers import register_error_handlers

        def create_app() -> Flask:
            app = Flask(__name__)
            ...
            register_error_handlers(app)
            return app
    """

    # -- Our own, deliberately-raised errors ---------------------------------
    @app.errorhandler(APIError)
    def handle_api_error(exc: APIError) -> Response:
        # 5xx is a server fault worth a traceback; 4xx is a client issue.
        if exc.status_code >= 500:
            logger.opt(exception=exc).error(
                "APIError {} ({}) on {}", exc.code, exc.status_code, _client_context()
            )
        else:
            logger.warning(
                "APIError {} ({}) on {}: {}",
                exc.code,
                exc.status_code,
                _client_context(),
                exc.message,
            )
        return _make_response(exc.to_dict(), exc.status_code)

    # -- Pydantic request/response validation --------------------------------
    if PydanticValidationError is not None:

        @app.errorhandler(PydanticValidationError)      
        def handle_validation_error(exc: PydanticValidationError) -> Response:       # type: ignore[arg-type]
            # Flatten Pydantic's error list into compact, client-safe details.
            details = [
                {
                    "field": ".".join(str(loc) for loc in err.get("loc", ())) or "(body)",
                    "message": err.get("msg", "invalid value"),
                    "type": err.get("type", "value_error"),
                }
                for err in exc.errors()
            ]
            api_error = InvalidDomainError(
                message="One or more fields failed validation.",
                code="VALIDATION_ERROR",
                details=details,
            )
            logger.warning(
                "Validation failed on {}: {} issue(s)",
                _client_context(),
                len(details),
            )
            return _make_response(api_error.to_dict(), api_error.status_code)

    # -- Flask-Limiter rate limiting -----------------------------------------
    if RateLimitExceeded is not None:

        @app.errorhandler(RateLimitExceeded)
        def handle_rate_limit(exc: RateLimitExceeded) -> Response:   # type: ignore[arg-type]
            # exc.description looks like "60 per 1 minute"; surface it to the caller.
            limit_desc = getattr(exc, "description", None)
            message = RateLimitError.message
            if limit_desc:
                message = f"Rate limit exceeded ({limit_desc}). Please slow down and try again shortly."
            api_error = RateLimitError(message=message)
            logger.warning("Rate limit hit on {}: {}", _client_context(), limit_desc)
            response = _make_response(api_error.to_dict(), api_error.status_code)
            # Preserve any Retry-After / rate-limit headers Flask-Limiter attached.
            retry_after = getattr(exc, "retry_after", None)
            if retry_after is not None:
                response.headers["Retry-After"] = str(retry_after)
            return response

    # -- Any other Werkzeug HTTP error (404, 405, 413, 415, ...) --------------
    @app.errorhandler(HTTPException)
    def handle_http_exception(exc: HTTPException) -> Response:
        status = exc.code or 500
        # Map the HTTP status onto a stable ADIS error code.
        code = {
            400: "BAD_REQUEST",
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            406: "NOT_ACCEPTABLE",
            413: "PAYLOAD_TOO_LARGE",
            415: "UNSUPPORTED_MEDIA_TYPE",
            429: "RATE_LIMIT_EXCEEDED",
        }.get(status, "HTTP_ERROR")

        # Werkzeug's default descriptions are user-safe for 4xx; scrub 5xx.
        message = exc.description if status < 500 else InternalServerError.message
        payload = {"error": {"code": code, "message": message, "status": status}}

        if status >= 500:
            logger.opt(exception=exc).error("HTTP {} on {}", status, _client_context())
        else:
            logger.warning("HTTP {} on {}: {}", status, _client_context(), exc.description)
        return _make_response(payload, status)

    # -- The catch-all: anything we didn't anticipate ------------------------
    @app.errorhandler(Exception)
    def handle_unexpected_error(exc: Exception) -> Response:
        # Never leak the real exception to the client — log it, return generic 500.
        logger.opt(exception=exc).error(
            "Unhandled {} on {}", type(exc).__name__, _client_context()
        )
        api_error = InternalServerError()
        return _make_response(api_error.to_dict(), api_error.status_code)

    logger.debug("ADIS error handlers registered.")
