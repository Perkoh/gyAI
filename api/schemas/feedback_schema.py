"""
api/schemas/feedback_schema.py
================================================================================
Pydantic v2 request/response models for the ADIS user-feedback endpoint.

Covers (Section 10.1) ``POST /api/v1/feedback`` — users (or the browser
extension's "Report This Site" action) report false positives / confirmed
malicious domains so the model can be retrained.

The request maps directly onto the ``user_feedback`` Supabase table (Section 9)::

    CREATE TABLE user_feedback (
        id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        domain        TEXT NOT NULL,
        system_label  TEXT NOT NULL,   -- what ADIS predicted
        user_verdict  TEXT NOT NULL,   -- 'false_positive'|'confirmed_malicious'|'unsure'
        user_comment  TEXT,
        created_at    TIMESTAMPTZ DEFAULT NOW()
    );

Domain normalization and the ``safe|suspicious|malicious`` label enum are reused
from ``analyze_schema`` so there is a single source of truth across the API.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from api.schemas.analyze_schema import DomainLabel, normalize_domain

__all__ = [
    "UserVerdict",
    "FeedbackRequest",
    "FeedbackResponse",
    "MAX_COMMENT_LENGTH",
]

#: Upper bound on a free-text user comment, to protect the database column.
MAX_COMMENT_LENGTH: int = 1000


# =============================================================================
# Enums
# =============================================================================
class UserVerdict(str, Enum):
    """The user's assessment of an ADIS verdict (``user_verdict`` column)."""

    FALSE_POSITIVE = "false_positive"          # ADIS flagged it, but it's safe.
    CONFIRMED_MALICIOUS = "confirmed_malicious"  # ADIS was right / user confirms bad.
    UNSURE = "unsure"                          # User isn't certain.


# =============================================================================
# Request model
# =============================================================================
class FeedbackRequest(BaseModel):
    """Body for ``POST /api/v1/feedback``."""

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "domain": "secure-login-paypa1.xyz",
                "system_label": "malicious",
                "user_verdict": "confirmed_malicious",
                "user_comment": "This mimicked my bank's login page.",
            }
        },
    )

    domain: str = Field(
        ...,
        description="The domain the feedback is about.",
        min_length=1,
        max_length=2048,
    )
    system_label: DomainLabel = Field(
        ...,
        description="The verdict ADIS originally gave: safe | suspicious | malicious.",
    )
    user_verdict: UserVerdict = Field(
        ...,
        description="The user's assessment: false_positive | confirmed_malicious | unsure.",
    )
    user_comment: str | None = Field(
        default=None,
        description="Optional free-text comment from the user.",
        max_length=MAX_COMMENT_LENGTH,
    )

    @field_validator("domain", mode="before")
    @classmethod
    def _normalize_domain(cls, v: object) -> str:
        return normalize_domain(v)

    @field_validator("user_comment", mode="before")
    @classmethod
    def _clean_comment(cls, v: object) -> str | None:
        # Treat empty/whitespace-only comments as "no comment".
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("user_comment must be a string")
        stripped = v.strip()
        return stripped or None


# =============================================================================
# Response model
# =============================================================================
class FeedbackResponse(BaseModel):
    """
    Confirmation returned after feedback is recorded.

    ``id`` and ``created_at`` are populated from the inserted Supabase row; they
    may be ``null`` if logging is disabled or the insert result is unavailable,
    in which case the feedback is still acknowledged.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "a3f2b1c4-0000-0000-0000-000000000000",
                "domain": "secure-login-paypa1.xyz",
                "status": "received",
                "message": "Thank you — your feedback has been recorded.",
                "created_at": "2026-07-13T19:31:00Z",
            }
        }
    )

    id: str | None = Field(default=None, description="Supabase row id for the feedback record.")
    domain: str = Field(..., description="The domain the feedback was recorded for.")
    status: str = Field(default="received", description="Machine-readable acknowledgement status.")
    message: str = Field(
        default="Thank you — your feedback has been recorded.",
        description="Human-readable acknowledgement.",
    )
    created_at: datetime | None = Field(
        default=None, description="When the feedback row was created (ISO-8601)."
    )

    @field_serializer("created_at")
    def _serialize_created_at(self, value: datetime | None) -> str | None:
        # Emit ISO-8601 regardless of whether the route dumps in python or json mode.
        return value.isoformat() if value is not None else None
