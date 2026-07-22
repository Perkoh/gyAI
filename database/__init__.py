"""
database/
─────────
Supabase PostgreSQL logging layer for ADIS.

Exports:
    db  — singleton SupabaseClient instance

    DomainAnalysisRow   — typed row for domain_analyses table
    FeedbackRow         — typed row for user_feedback table
    ModelVersionRow     — typed row for model_versions table
    KnownDomainRow      — typed row for known_domains table
    AnalysisStats       — aggregate stats dataclass

Quick import:
    from database import db
    from database.supabase_client import db, ModelVersionRow
"""

from database.supabase_client import (
    db,
    SupabaseClient,
    DomainAnalysisRow,
    FeedbackRow,
    ModelVersionRow,
    KnownDomainRow,
    AnalysisStats,
)

__all__ = [
    "db",
    "SupabaseClient",
    "DomainAnalysisRow",
    "FeedbackRow",
    "ModelVersionRow",
    "KnownDomainRow",
    "AnalysisStats",
]
