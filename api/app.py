"""
gyAI — AI-Powered Domain Intelligence System (ADIS)
api/app.py

Flask application factory (blueprint section 13: "app.py — Flask application
factory (create_app)").

Responsibilities:
  * Load configuration (config/settings.py -> env vars -> defaults).
  * Initialise shared extensions (CORS, rate limiter, Redis) via
    api/extensions.py.
  * Register global error handlers producing the blueprint's error contract:
        {"error": {"code": ..., "message": ..., "status": ...}}
  * Register the v1 route blueprints (analyze, feedback, health, admin).
  * Warm the LightGBM model into memory once at startup (blueprint section 5.2).
  * Apply ProxyFix so client IPs are correct behind Fly.io / Nginx (needed for
    per-IP rate limiting, blueprint section 4.1).

Run in production with Gunicorn using the factory pattern (4 workers per the
blueprint):

    gunicorn -w 4 -b 0.0.0.0:8080 "api.app:create_app()"

Route modules, middleware, and config are imported defensively: modules that
have not been built yet are skipped with a warning so the API can boot and
serve /health during incremental development, rather than failing to import.
Genuine errors inside an existing module are NOT swallowed.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, Flask, jsonify, request
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

from api.extensions import init_extensions, limiter, redis_healthy

try:
    from loguru import logger
except Exception:  # pragma: no cover - degrade if loguru absent
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("adis.app")  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Env keys copied into app.config when present. Values are read as strings and
# coerced where noted in _coerce_config().
_ENV_KEYS = (
    "REDIS_URL", "REDIS_SOCKET_TIMEOUT",
    "RATELIMIT_STORAGE_URI",
    "SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_SERVICE_KEY",
    "API_KEY", "ADMIN_API_KEY",
    "MODEL_PATH", "MODEL_VERSION",
    "LOG_LEVEL", "FLASK_ENV", "ENV",
    "REQUIRE_MODEL", "RATELIMIT_ENABLED",
)

_DEFAULT_CONFIG: Dict[str, Any] = {
    "ENV": os.getenv("FLASK_ENV", os.getenv("ENV", "production")),
    "JSON_SORT_KEYS": False,
    "REDIS_URL": "redis://localhost:6379/0",
    "REDIS_SOCKET_TIMEOUT": 2.0,
    "MODEL_PATH": "ml/models/lgbm_model_v1.1.0.pkl",
    "MODEL_VERSION": None,        # resolved from the filename if left unset
    "REQUIRE_MODEL": False,       # if True, boot fails when the model is missing
    "LOG_LEVEL": "INFO",
    "MODEL_LOADED": False,
}

# (module_path, default_url_prefix). Route modules are expected to define a
# Blueprint (named `bp`, `blueprint`, or `<something>_bp`).
_BLUEPRINT_REGISTRY: Tuple[Tuple[str, str], ...] = (
    ("api.routes.health", "/api/v1"),
    ("api.routes.analyze", "/api/v1"),
    ("api.routes.feedback", "/api/v1"),
    ("api.routes.admin", "/api/v1/admin"),
)


def _coerce_config(app: Flask) -> None:
    """Coerce known string env values into their proper types."""
    for bool_key in ("REQUIRE_MODEL", "RATELIMIT_ENABLED"):
        val = app.config.get(bool_key)
        if isinstance(val, str):
            app.config[bool_key] = val.strip().lower() in ("1", "true", "yes", "on")
    tv = app.config.get("REDIS_SOCKET_TIMEOUT")
    if isinstance(tv, str):
        try:
            app.config["REDIS_SOCKET_TIMEOUT"] = float(tv)
        except ValueError:
            app.config["REDIS_SOCKET_TIMEOUT"] = 2.0


def _load_config(app: Flask, overrides: Optional[Dict[str, Any]]) -> None:
    app.config.update(_DEFAULT_CONFIG)

    # 1) config/settings.py (authoritative when present): copy UPPERCASE attrs.
    try:  # pragma: no cover - depends on config/settings.py existing
        from config import settings  # type: ignore

        for key in dir(settings):
            if key.isupper():
                app.config[key] = getattr(settings, key)
        logger.info("Loaded configuration from config.settings.")
    except Exception as exc:
        logger.debug(f"config.settings not loaded ({exc}); using env/defaults.")

    # 2) Environment overlay for known keys.
    for key in _ENV_KEYS:
        env_val = os.getenv(key)
        if env_val is not None:
            app.config[key] = env_val

    # 3) Explicit overrides (used by tests).
    if overrides:
        app.config.update(overrides)

    _coerce_config(app)


# --------------------------------------------------------------------------- #
# Error handling (fallback contract; api/middleware/error_handlers.py may
# override once built).
# --------------------------------------------------------------------------- #

_STATUS_CODE_NAMES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    422: "UNPROCESSABLE_ENTITY",
    429: "RATE_LIMITED",
    500: "INTERNAL_ERROR",
    503: "SERVICE_UNAVAILABLE",
}


def _error_response(status: int, message: str, code: Optional[str] = None):
    code = code or _STATUS_CODE_NAMES.get(status, "ERROR")
    return (
        jsonify({"error": {"code": code, "message": message, "status": status}}),
        status,
    )


def _install_fallback_error_handlers(app: Flask) -> None:
    @app.errorhandler(HTTPException)
    def _handle_http_exception(exc: HTTPException):
        status = exc.code or 500
        message = exc.description or exc.name or "Request failed"
        return _error_response(status, message)

    @app.errorhandler(429)
    def _handle_rate_limit(exc: Any):
        desc = getattr(exc, "description", None) or "Rate limit exceeded"
        return _error_response(429, str(desc), code="RATE_LIMITED")

    @app.errorhandler(Exception)
    def _handle_unexpected(exc: Exception):
        # Log the full traceback; never leak internals to the client.
        logger.exception(f"Unhandled exception: {exc}")
        return _error_response(500, "An internal error occurred.")

    logger.info("Installed fallback error handlers.")


def _register_error_handlers(app: Flask) -> None:
    try:
        from api.middleware.error_handlers import register_error_handlers  # type: ignore

        register_error_handlers(app)
        logger.info("Registered error handlers from api.middleware.error_handlers.")
    except ModuleNotFoundError:
        _install_fallback_error_handlers(app)
    except Exception as exc:
        logger.warning(
            f"error_handlers module present but failed ({exc}); using fallback."
        )
        _install_fallback_error_handlers(app)


# --------------------------------------------------------------------------- #
# Blueprint registration
# --------------------------------------------------------------------------- #

def _find_blueprint(module: Any) -> Optional[Blueprint]:
    for name in ("bp", "blueprint"):
        obj = getattr(module, name, None)
        if isinstance(obj, Blueprint):
            return obj
    for value in vars(module).values():
        if isinstance(value, Blueprint):
            return value
    return None


def _register_blueprints(app: Flask) -> None:
    registered: List[str] = []
    for module_path, default_prefix in _BLUEPRINT_REGISTRY:
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError:
            logger.info(f"Route module '{module_path}' not present yet; skipping.")
            continue
        except Exception:
            logger.exception(f"Failed importing route module '{module_path}'.")
            continue

        bp = _find_blueprint(module)
        if bp is None:
            logger.warning(f"No Blueprint found in '{module_path}'; skipping.")
            continue

        # Respect a prefix the blueprint set for itself; otherwise apply ours.
        if bp.url_prefix:
            app.register_blueprint(bp)
            prefix = bp.url_prefix
        else:
            app.register_blueprint(bp, url_prefix=default_prefix)
            prefix = default_prefix
        registered.append(f"{bp.name}@{prefix}")

    if registered:
        logger.info(f"Registered blueprints: {', '.join(registered)}")
    else:
        logger.warning("No route blueprints registered (none present yet).")


# --------------------------------------------------------------------------- #
# Model warm-up
# --------------------------------------------------------------------------- #

def _warm_model(app: Flask) -> None:
    try:
        from ml.model_server import get_model_server  # type: ignore

        server = get_model_server(
            model_path=app.config.get("MODEL_PATH"),
            model_version=app.config.get("MODEL_VERSION"),
            auto_load=True,
        )
        app.extensions["model_server"] = server
        app.config["MODEL_LOADED"] = bool(server.is_loaded)
        app.config["MODEL_VERSION"] = server.model_version
        logger.info(
            f"Model warm: version={server.model_version}, loaded={server.is_loaded}"
        )
    except Exception as exc:
        app.config["MODEL_LOADED"] = False
        if app.config.get("REQUIRE_MODEL"):
            logger.error(f"Model failed to load and REQUIRE_MODEL is set: {exc}")
            raise
        logger.warning(
            f"Model not loaded at startup ({exc}); /health will report degraded."
        )


# --------------------------------------------------------------------------- #
# Root / infra routes (not part of the versioned API surface)
# --------------------------------------------------------------------------- #

def _register_root_routes(app: Flask) -> None:
    @app.get("/")
    @limiter.exempt
    def index():
        return jsonify(
            {
                "name": "gyAI — AI-Powered Domain Intelligence System (ADIS)",
                "status": "ok",
                "model_version": app.config.get("MODEL_VERSION"),
                "api_base": "/api/v1",
            }
        )

    @app.get("/health")
    @app.get("/healthz")
    @limiter.exempt
    def health():
        # Lightweight probe for Fly.io / load balancers. The richer
        # /api/v1/health lives in the health blueprint.
        model_ok = bool(app.config.get("MODEL_LOADED"))
        redis_ok = redis_healthy()
        status = "ok" if model_ok else "degraded"
        payload = {
            "status": status,
            "model_loaded": model_ok,
            "redis": "ok" if redis_ok else "unavailable",
            "model_version": app.config.get("MODEL_VERSION"),
        }
        return jsonify(payload), (200 if model_ok else 503)


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #

def create_app(config_overrides: Optional[Dict[str, Any]] = None) -> Flask:
    """
    Build and configure the ADIS Flask application.

    Parameters
    ----------
    config_overrides : optional dict merged into app.config last (for tests).
    """
    app = Flask(__name__)

    _load_config(app, config_overrides)

    # Correct client IPs behind the Fly.io / Nginx reverse proxy so per-IP
    # rate limiting and logging see the real address, not the proxy's.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)  # type: ignore[assignment]

    init_extensions(app)
    _register_error_handlers(app)
    _register_blueprints(app)
    _register_root_routes(app)
    _warm_model(app)

    logger.info(
        f"ADIS app created (env={app.config.get('ENV')}, "
        f"model_loaded={app.config.get('MODEL_LOADED')})."
    )
    return app


if __name__ == "__main__":
    # Local development server only. Use Gunicorn in production:
    #   gunicorn -w 4 -b 0.0.0.0:8080 "api.app:create_app()"
    application = create_app()
    port = int(os.getenv("PORT", "8080"))
    application.run(host="0.0.0.0", port=port, debug=True)
