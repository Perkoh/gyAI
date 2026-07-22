"""
api/schemas/analyze_schema.py
================================================================================
Pydantic v2 request/response models for the ADIS domain-analysis endpoints.

Covers (Section 10.2 of the ADIS blueprint):

    POST /api/v1/analyze         -> AnalyzeRequest  -> AnalyzeResponse
    POST /api/v1/analyze/bulk    -> BulkAnalyzeRequest -> BulkAnalyzeResponse

The response shape mirrors the blueprint exactly::

    {
        "domain": "secure-login-paypa1.xyz",
        "score": 0.9341,
        "label": "malicious",
        "confidence": "high",
        "reasons": ["...", "..."],
        "model_version": "v1.2.0",
        "analysis_id": "a3f2b1c4-...",   # null for safe domains (not logged)
        "duration_ms": 187,
        "cached": false,
        "network_features_used": true
    }

Serialization
-------------
These models are designed to be serialized with ``model_dump(mode="json")`` (or
``model_dump_json()``) so enums render as their string values and any datetimes
render as ISO-8601. Enum fields additionally use ``use_enum_values=True`` so they
serialize to plain strings even under ``model_dump()`` (python mode).

Validation
----------
The single source of truth for domain normalization/validation is
``normalize_domain`` — it is reused by the bulk request model here and by
``feedback_schema.py``. A validation failure raises ``ValueError`` inside the
Pydantic validator, which Pydantic wraps into a ``ValidationError``. The global
error handler (``api/middleware/error_handlers.py``) turns that into a 422 JSON
envelope. Routes that want the blueprint's exact ``INVALID_DOMAIN`` error code
for the single-domain endpoint may catch ``ValidationError`` and re-raise
``InvalidDomainError`` (see notes in ``routes/analyze.py``).
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "DomainLabel",
    "Confidence",
    "normalize_domain",
    "label_for_score",
    "confidence_for_score",
    "AnalyzeRequest",
    "AnalyzeResponse",
    "BulkAnalyzeRequest",
    "BulkAnalyzeResponse",
    "ErrorDetail",
    "ErrorResponse",
    "SAFE_THRESHOLD",
    "MALICIOUS_THRESHOLD",
    "MAX_BULK_DOMAINS",
]


# =============================================================================
# Enums
# =============================================================================
class DomainLabel(str, Enum):
    """The verdict ADIS assigns to a domain (Section 4.3 / 11.1)."""

    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"


class Confidence(str, Enum):
    """How confident the model is in its verdict, derived from the score."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# =============================================================================
# Scoring thresholds & derivations
# =============================================================================
#: score < 0.50  -> safe (silent).  Section 4.3 / 11.1.
SAFE_THRESHOLD: float = 0.50
#: score >= 0.80 -> malicious (red alert).  0.50–0.79 -> suspicious (yellow).
MALICIOUS_THRESHOLD: float = 0.80

#: Maximum domains accepted by POST /analyze/bulk (Section 10.1).
MAX_BULK_DOMAINS: int = 50


def label_for_score(score: float) -> DomainLabel:
    """Map a probability score in [0, 1] to a :class:`DomainLabel`."""
    if score >= MALICIOUS_THRESHOLD:
        return DomainLabel.MALICIOUS
    if score >= SAFE_THRESHOLD:
        return DomainLabel.SUSPICIOUS
    return DomainLabel.SAFE


def confidence_for_score(score: float) -> Confidence:
    """
    Derive a confidence band from the score's distance from the 0.5 boundary.

    Scores near 0 or 1 are confident; scores near the decision boundary (0.5)
    are not. e.g. 0.9341 -> high, 0.62 -> medium, 0.48 -> low.
    """
    if score <= 0.15 or score >= 0.85:
        return Confidence.HIGH
    if score <= 0.35 or score >= 0.65:
        return Confidence.MEDIUM
    return Confidence.LOW


# =============================================================================
# Domain normalization / validation  (single source of truth)
# =============================================================================
MAX_DOMAIN_LENGTH: int = 253  # RFC 1035 limit on a fully-qualified domain name.

# Strips an optional URI scheme (http://, https://, ftp://, ...).
_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)

# A single DNS label: 1–63 chars, alphanumerics (incl. IDN unicode) and internal
# hyphens, never starting or ending with a hyphen.
_LABEL_RE = re.compile(
    r"^(?!-)[a-z0-9\u00a1-\uffff](?:[a-z0-9\u00a1-\uffff-]{0,61}[a-z0-9\u00a1-\uffff])?$",
    re.IGNORECASE,
)

# A TLD: 2+ alphabetic/IDN chars, or an ``xn--`` punycode label.
_TLD_RE = re.compile(r"^(xn--[a-z0-9-]+|[a-z\u00a1-\uffff]{2,})$", re.IGNORECASE)

# Rough IPv4 detector, for a clearer error than "invalid TLD".
_IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


def normalize_domain(value: object) -> str:
    """
    Normalize and validate a raw domain string.

    Accepts a bare hostname (``example.com``) and is also lenient about
    developer-supplied URLs: a scheme, userinfo, port, path, query and fragment
    are stripped, and the result is lower-cased. Subdomains and a leading
    ``www.`` are preserved, because features such as ``has_www`` and
    ``num_subdomains`` depend on them.

    Returns the cleaned domain, or raises ``ValueError`` describing why the input
    is not a usable domain name.
    """
    if not isinstance(value, str):
        raise ValueError("domain must be a string")

    domain = value.strip().lower()
    if not domain:
        raise ValueError("domain must not be empty")

    domain = _SCHEME_RE.sub("", domain)          # drop scheme://
    if "@" in domain:                            # drop user:pass@
        domain = domain.rsplit("@", 1)[-1]
    for sep in ("/", "?", "#"):                  # drop path/query/fragment
        if sep in domain:
            domain = domain.split(sep, 1)[0]
    if ":" in domain:                            # drop :port
        domain = domain.split(":", 1)[0]
    domain = domain.strip().rstrip(".")          # drop FQDN trailing dot

    if not domain:
        raise ValueError("domain must not be empty after normalization")
    if len(domain) > MAX_DOMAIN_LENGTH:
        raise ValueError(f"domain exceeds the maximum length of {MAX_DOMAIN_LENGTH} characters")
    if _IPV4_RE.match(domain):
        raise ValueError("IP addresses are not supported; provide a domain name")
    if "." not in domain:
        raise ValueError("domain must contain at least one dot (e.g. 'example.com')")

    labels = domain.split(".")
    for label in labels:
        if not label:
            raise ValueError("domain contains an empty label (consecutive dots)")
        if not _LABEL_RE.match(label):
            raise ValueError(f"'{label}' is not a valid domain label")

    if not _TLD_RE.match(labels[-1]):
        raise ValueError("domain must end with a valid top-level domain")

    return domain


# =============================================================================
# Request models
# =============================================================================
class AnalyzeRequest(BaseModel):
    """Body for ``POST /api/v1/analyze``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"domain": "secure-login-paypa1.xyz"}},
    )

    domain: str = Field(
        ...,
        description="The domain (hostname) to analyze. No scheme or path required.",
        min_length=1,
        max_length=2048,  # generous outer bound before normalization trims it
    )

    @field_validator("domain", mode="before")
    @classmethod
    def _normalize(cls, v: object) -> str:
        return normalize_domain(v)


class BulkAnalyzeRequest(BaseModel):
    """Body for ``POST /api/v1/analyze/bulk`` (authenticated). Up to 50 domains."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {"domains": ["github.com", "secure-login-paypa1.xyz"]}
        },
    )

    domains: list[str] = Field(
        ...,
        description=f"1–{MAX_BULK_DOMAINS} domains to analyze in a single request.",
        min_length=1,
        max_length=MAX_BULK_DOMAINS,
    )

    @field_validator("domains", mode="before")
    @classmethod
    def _normalize_each(cls, v: object) -> list[str]:
        if not isinstance(v, (list, tuple)):
            raise ValueError("domains must be a list")
        if len(v) == 0:
            raise ValueError("domains must contain at least one entry")
        if len(v) > MAX_BULK_DOMAINS:
            raise ValueError(f"a maximum of {MAX_BULK_DOMAINS} domains may be submitted at once")
        # Normalize each, then de-duplicate while preserving first-seen order.
        seen: set[str] = set()
        cleaned: list[str] = []
        for item in v:
            norm = normalize_domain(item)
            if norm not in seen:
                seen.add(norm)
                cleaned.append(norm)
        return cleaned


# =============================================================================
# Response models
# =============================================================================
class AnalyzeResponse(BaseModel):
    """
    Result of analyzing a single domain (``POST /api/v1/analyze``).

    ``analysis_id`` is the Supabase audit-log id; it is ``null`` for safe domains
    because safe domains are not logged (blueprint FLAG 1).
    """

    # ``protected_namespaces=()`` allows the ``model_version`` field name, which
    # would otherwise collide with Pydantic's reserved ``model_`` namespace.
    model_config = ConfigDict(
        use_enum_values=True,
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "domain": "secure-login-paypa1.xyz",
                "score": 0.9341,
                "label": "malicious",
                "confidence": "high",
                "reasons": [
                    "The domain was registered only 4 days ago",
                    "The domain name contains phishing keywords: 'login', 'secure'",
                    "The domain uses a high-risk extension (.xyz)",
                ],
                "model_version": "v1.2.0",
                "analysis_id": "a3f2b1c4-0000-0000-0000-000000000000",
                "duration_ms": 187,
                "cached": False,
                "network_features_used": True,
            }
        },
    )

    domain: str = Field(..., description="The normalized domain that was analyzed.")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability the domain is malicious/phishing, in [0, 1].",
    )
    label: DomainLabel = Field(..., description="safe | suspicious | malicious.")
    confidence: Confidence = Field(..., description="low | medium | high.")
    reasons: list[str] = Field(
        default_factory=list,
        description="Human-readable SHAP-derived reasons. Empty for safe domains.",
    )
    model_version: str = Field(..., description="Version of the model that produced this result.")
    analysis_id: str | None = Field(
        default=None,
        description="Supabase audit-log id; null when the domain was not logged (safe).",
    )
    duration_ms: int = Field(..., ge=0, description="Server-side analysis time in milliseconds.")
    cached: bool = Field(..., description="Whether this result was served from the Redis cache.")
    network_features_used: bool = Field(
        ...,
        description="Whether live DNS/WHOIS features were available for this analysis.",
    )

    @field_validator("score", mode="after")
    @classmethod
    def _round_score(cls, v: float) -> float:
        # Match the blueprint's 4-decimal presentation (e.g. 0.9341).
        return round(v, 4)

    @classmethod
    def from_prediction(
        cls,
        *,
        domain: str,
        score: float,
        model_version: str,
        reasons: list[str] | None = None,
        analysis_id: str | None = None,
        duration_ms: int = 0,
        cached: bool = False,
        network_features_used: bool = False,
    ) -> "AnalyzeResponse":
        """
        Build a response from a raw model score, deriving ``label`` and
        ``confidence`` from the blueprint thresholds. Reasons are only meaningful
        for non-safe domains, so they are dropped when the verdict is ``safe``.
        """
        label = label_for_score(score)
        supplied_reasons = reasons or []
        return cls(
            domain=domain,
            score=score,
            label=label,
            confidence=confidence_for_score(score),
            reasons=[] if label == DomainLabel.SAFE else supplied_reasons,
            model_version=model_version,
            analysis_id=analysis_id,
            duration_ms=duration_ms,
            cached=cached,
            network_features_used=network_features_used,
        )


class BulkAnalyzeResponse(BaseModel):
    """Result of ``POST /api/v1/analyze/bulk`` — one entry per unique domain."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "count": 2,
                "duration_ms": 402,
                "results": [
                    {
                        "domain": "github.com",
                        "score": 0.0213,
                        "label": "safe",
                        "confidence": "high",
                        "reasons": [],
                        "model_version": "v1.2.0",
                        "analysis_id": None,
                        "duration_ms": 9,
                        "cached": True,
                        "network_features_used": False,
                    },
                    {
                        "domain": "secure-login-paypa1.xyz",
                        "score": 0.9341,
                        "label": "malicious",
                        "confidence": "high",
                        "reasons": ["The domain uses a high-risk extension (.xyz)"],
                        "model_version": "v1.2.0",
                        "analysis_id": "a3f2b1c4-0000-0000-0000-000000000000",
                        "duration_ms": 190,
                        "cached": False,
                        "network_features_used": True,
                    },
                ],
            }
        }
    )

    count: int = Field(..., ge=0, description="Number of unique domains analyzed.")
    duration_ms: int = Field(..., ge=0, description="Total server-side time for the batch.")
    results: list[AnalyzeResponse] = Field(
        default_factory=list, description="Per-domain analysis results."
    )

    @model_validator(mode="after")
    def _check_count(self) -> "BulkAnalyzeResponse":
        if self.count != len(self.results):
            raise ValueError("count must equal the number of results")
        return self


# =============================================================================
# Error contract (documentation mirror of error_handlers.py)
# =============================================================================
class ErrorDetail(BaseModel):
    """The inner object of an ADIS error response."""

    code: str = Field(..., description="Stable machine-readable error code.")
    message: str = Field(..., description="Human-readable explanation.")
    status: int = Field(..., description="HTTP status code.")


class ErrorResponse(BaseModel):
    """
    The ADIS error envelope (Section 10.2).

    Note: this model documents the contract for API consumers and tests. At
    runtime, error responses are produced by ``api/middleware/error_handlers.py``,
    which is the authoritative source.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "code": "INVALID_DOMAIN",
                    "message": "The provided value is not a valid domain name",
                    "status": 422,
                }
            }
        }
    )

    error: ErrorDetail
