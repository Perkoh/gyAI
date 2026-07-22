"""
api/routes/feedback.py
================================================================================
Public feedback endpoint (blueprint section 10.1):

    POST /api/v1/feedback    Submit a false-positive / confirmed-malicious
                             report. RATE_LIMIT_FEEDBACK (per IP).

The body is validated by ``FeedbackRequest`` and written synchronously via
``db.insert_feedback`` into the ``user_feedback`` table. The client returns
only a success boolean (no row id / timestamp), so the acknowledgement carries
``id: null`` and distinguishes outcomes by status:

    "received"  — the row was written to Supabase (HTTP 201)
    "accepted"  — validation passed but the write could not be confirmed
                  (database down / unconfigured); the report is acknowledged
                  so the extension's "Report This Site" flow never appears to
                  fail (HTTP 202)
"""

from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify
from loguru import logger
from pydantic import ValidationError

from api.extensions import RATE_LIMIT_FEEDBACK, limiter
from api.middleware.error_handlers import InvalidDomainError
from api.routes import first_error_message, require_json_object
from api.schemas.feedback_schema import FeedbackRequest, FeedbackResponse
from config.settings import settings
from database.supabase_client import db

feedback_bp = Blueprint("feedback", __name__, url_prefix=settings.API_PREFIX)


@feedback_bp.post("/feedback")
@limiter.limit(RATE_LIMIT_FEEDBACK)
def submit_feedback() -> Any:
    """Record a user's verdict on a previously-analyzed domain."""
    data = require_json_object()
    try:
        req = FeedbackRequest.model_validate(data)
    except ValidationError as exc:
        raise InvalidDomainError(first_error_message(exc))

    ok = False
    try:
        ok = db.insert_feedback(
            domain=req.domain,
            system_label=req.system_label,
            user_verdict=req.user_verdict,
            user_comment=req.user_comment,
        )
    except Exception as exc:  # the client is fault-tolerant; belt & braces
        logger.warning("Feedback insert raised for '{}': {}", req.domain, exc)

    if ok:
        logger.info("Feedback recorded for '{}' ({})", req.domain, req.user_verdict)
        status, http_status = "received", 201
        message = "Thank you — your feedback has been recorded."
    else:
        logger.warning("Feedback for '{}' accepted but not persisted", req.domain)
        status, http_status = "accepted", 202
        message = "Thank you — your feedback has been accepted."

    response = FeedbackResponse(
        id=None,  # db.insert_feedback returns only a boolean, never the row id
        domain=req.domain,
        status=status,
        message=message,
        created_at=None,
    )
    return jsonify(response.model_dump(mode="json")), http_status
