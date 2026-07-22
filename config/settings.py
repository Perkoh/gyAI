"""
config/settings.py
──────────────────
Centralised configuration for the ADIS API.

All tunable values live here. Nothing reads os.environ directly outside
this file — every other module imports from here.

Usage:
    from config.settings import settings

    redis_url = settings.REDIS_URL
    threshold = settings.SCORE_THRESHOLD_ALERT
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────
# Load .env (has no effect when vars are already set, e.g. in
# Fly.io / Docker production environments)
# ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent  # adis/
load_dotenv(_ROOT / ".env")


def _require(key: str) -> str:
    """Return an env var or raise a clear error at startup."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Add it to your .env file or deployment secrets."
        )
    return value


def _bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).strip().lower() in ("1", "true", "yes")


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _list(key: str, default: str = "") -> List[str]:
    """Comma-separated env var → Python list, stripping whitespace."""
    raw = os.getenv(key, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# ══════════════════════════════════════════════════════════════
# Settings class — instantiated once at the bottom of this file
# ══════════════════════════════════════════════════════════════

class _Settings:
    # ── Environment ──────────────────────────────────────────
    ENV: str = os.getenv("FLASK_ENV", "production")
    DEBUG: bool = _bool("FLASK_DEBUG", default=False)
    TESTING: bool = _bool("TESTING", default=False)

    # ── Flask Core ───────────────────────────────────────────
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production")
    # Prefix for all API routes: /api/v1
    API_PREFIX: str = "/api/v1"
    API_VERSION: str = "v1"

    # ── Gunicorn / Server ────────────────────────────────────
    # Used by the Dockerfile entrypoint; not read by Flask itself
    GUNICORN_WORKERS: int = _int("GUNICORN_WORKERS", default=4)
    GUNICORN_TIMEOUT: int = _int("GUNICORN_TIMEOUT", default=10)   # seconds
    PORT: int = _int("PORT", default=8080)
    HOST: str = os.getenv("HOST", "0.0.0.0")

    # ── Redis / Upstash ──────────────────────────────────────
    # Full redis:// or rediss:// URL.  Upstash supplies this.
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    # Namespace prefix for every cache key
    REDIS_KEY_PREFIX: str = "adis:cache:"
    # TTL for safe domains (score < 0.50) — 1 hour
    REDIS_TTL_SAFE: int = _int("REDIS_TTL_SAFE", default=3600)
    # TTL for suspicious domains (0.50 ≤ score < 0.80) — 15 minutes
    REDIS_TTL_SUSPICIOUS: int = _int("REDIS_TTL_SUSPICIOUS", default=900)
    # TTL for malicious domains (score ≥ 0.80) — 15 minutes
    REDIS_TTL_MALICIOUS: int = _int("REDIS_TTL_MALICIOUS", default=900)
    # Socket connect/read timeout for Redis commands (seconds)
    REDIS_SOCKET_TIMEOUT: float = _float("REDIS_SOCKET_TIMEOUT", default=2.0)
    REDIS_SOCKET_CONNECT_TIMEOUT: float = _float(
        "REDIS_SOCKET_CONNECT_TIMEOUT", default=2.0
    )
    # Whether to use SSL for Redis (Upstash requires True in production)
    REDIS_SSL: bool = _bool("REDIS_SSL", default=False)

    # ── Supabase ─────────────────────────────────────────────
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    # Only log domains at or above this score to Supabase
    # (matches SCORE_THRESHOLD_CAUTION — skip safe domains)
    SUPABASE_LOG_THRESHOLD: float = _float("SUPABASE_LOG_THRESHOLD", default=0.50)

    # ── LightGBM Model ───────────────────────────────────────
    MODEL_DIR: Path = _ROOT / "ml" / "models"
    # Active model filename — symlink or explicit name
    MODEL_FILENAME: str = os.getenv("MODEL_FILENAME", "lgbm_model_v1.1.0.pkl")
    MODEL_PATH: Path = MODEL_DIR / MODEL_FILENAME
    # Semantic version string returned in API responses
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "v1.1.0")
    # Number of top SHAP reasons to include in the response (2 for caution, 3 for alert)
    SHAP_MAX_REASONS: int = _int("SHAP_MAX_REASONS", default=3)

    # ── Scoring Thresholds ───────────────────────────────────
    # score < CAUTION  → label = "safe",       no notification
    # CAUTION ≤ score < ALERT → label = "suspicious", yellow banner
    # score ≥ ALERT    → label = "malicious",  red banner
    SCORE_THRESHOLD_CAUTION: float = _float("SCORE_THRESHOLD_CAUTION", default=0.50)
    SCORE_THRESHOLD_ALERT: float = _float("SCORE_THRESHOLD_ALERT", default=0.80)

    # Confidence bucket boundaries (applied to the raw probability score)
    # score < LOW_CONF_CUTOFF       → confidence = "low"
    # LOW_CONF_CUTOFF ≤ score < HIGH → confidence = "medium"
    # score ≥ HIGH_CONF_CUTOFF      → confidence = "high"
    CONFIDENCE_LOW_CUTOFF: float = _float("CONFIDENCE_LOW_CUTOFF", default=0.65)
    CONFIDENCE_HIGH_CUTOFF: float = _float("CONFIDENCE_HIGH_CUTOFF", default=0.85)

    # ── Feature Extraction ───────────────────────────────────
    # Hard timeout for WHOIS lookups (seconds)
    WHOIS_TIMEOUT: int = _int("WHOIS_TIMEOUT", default=3)
    # Hard timeout for DNS lookups (seconds)
    DNS_TIMEOUT: float = _float("DNS_TIMEOUT", default=3.0)
    # Total number of features in the model input vector
    FEATURE_COUNT: int = 48

    # ── API Rate Limits (flask-limiter strings) ───────────────
    # POST /analyze — public
    RATE_LIMIT_ANALYZE: str = os.getenv("RATE_LIMIT_ANALYZE", "60 per minute")
    # POST /analyze/bulk — authenticated
    RATE_LIMIT_BULK: str = os.getenv("RATE_LIMIT_BULK", "10 per minute")
    # POST /feedback — public
    RATE_LIMIT_FEEDBACK: str = os.getenv("RATE_LIMIT_FEEDBACK", "20 per minute")
    # All other endpoints — default
    RATE_LIMIT_DEFAULT: str = os.getenv("RATE_LIMIT_DEFAULT", "100 per minute")

    # ── CORS ─────────────────────────────────────────────────
    # Allowed origins for browser extension requests.
    # In production, restrict to chrome-extension:// and moz-extension://.
    # In development, "*" is acceptable.
    CORS_ORIGINS: List[str] = _list(
        "CORS_ORIGINS",
        default="chrome-extension://*,moz-extension://*",
    )
    CORS_SUPPORTS_CREDENTIALS: bool = _bool("CORS_SUPPORTS_CREDENTIALS", default=False)

    # ── API Key (for protected/admin endpoints) ───────────────
    # Generate with: python -c "import secrets; print(secrets.token_hex(32))"
    API_KEY: str = os.getenv("ADIS_API_KEY", "")
    ADMIN_KEY: str = os.getenv("ADIS_ADMIN_KEY", "")

    # ── Logging (loguru) ─────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_JSON: bool = _bool("LOG_JSON", default=False)  # True in production
    LOG_FILE: str = os.getenv("LOG_FILE", "")          # "" = stdout only

    # ── Bulk Analyze ─────────────────────────────────────────
    BULK_MAX_DOMAINS: int = _int("BULK_MAX_DOMAINS", default=50)

    # ── Notification / Extension Behaviour ───────────────────
    # Seconds before caution (yellow) banner auto-dismisses
    CAUTION_AUTO_DISMISS_SECONDS: int = _int("CAUTION_AUTO_DISMISS_SECONDS", default=8)

    # ── Fly.io / Production deployment ───────────────────────
    FLY_APP_NAME: str = os.getenv("FLY_APP_NAME", "gyai-api")
    BASE_URL: str = os.getenv(
        "BASE_URL", f"https://{os.getenv('FLY_APP_NAME', 'gyai-api')}.fly.dev"
    )

    # ── Derived helpers ───────────────────────────────────────

    @property
    def is_development(self) -> bool:
        return self.ENV in ("development", "dev")

    @property
    def is_production(self) -> bool:
        return self.ENV == "production"

    @property
    def is_testing(self) -> bool:
        return self.TESTING

    @property
    def supabase_configured(self) -> bool:
        """True if both Supabase credentials are present."""
        return bool(self.SUPABASE_URL and self.SUPABASE_KEY)

    @property
    def redis_configured(self) -> bool:
        """True if a non-default Redis URL is configured."""
        return self.REDIS_URL not in ("", "redis://localhost:6379/0") or self.is_development

    def score_to_label(self, score: float) -> str:
        """Map a raw probability score to a human-readable label string."""
        if score >= self.SCORE_THRESHOLD_ALERT:
            return "malicious"
        if score >= self.SCORE_THRESHOLD_CAUTION:
            return "suspicious"
        return "safe"

    def score_to_confidence(self, score: float) -> str:
        """Map a raw probability score to a confidence bucket string."""
        if score >= self.CONFIDENCE_HIGH_CUTOFF or score <= (1 - self.CONFIDENCE_HIGH_CUTOFF):
            return "high"
        if score >= self.CONFIDENCE_LOW_CUTOFF or score <= (1 - self.CONFIDENCE_LOW_CUTOFF):
            return "medium"
        return "low"

    def cache_ttl_for_label(self, label: str) -> int:
        """Return the Redis TTL (seconds) appropriate for a given label."""
        if label == "safe":
            return self.REDIS_TTL_SAFE
        if label == "suspicious":
            return self.REDIS_TTL_SUSPICIOUS
        return self.REDIS_TTL_MALICIOUS  # "malicious"

    def redis_key(self, domain: str) -> str:
        """Construct the namespaced Redis cache key for a domain."""
        return f"{self.REDIS_KEY_PREFIX}{domain.lower()}"

    def __repr__(self) -> str:
        return (
            f"<ADISSettings env={self.ENV} debug={self.DEBUG} "
            f"model={self.MODEL_VERSION}>"
        )


# ── Singleton ─────────────────────────────────────────────────
settings = _Settings()
