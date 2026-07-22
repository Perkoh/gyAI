"""
database/supabase_client.py
───────────────────────────
Supabase PostgreSQL logging client for ADIS.

Responsibilities
────────────────
  • Log suspicious / malicious domain analyses to ``domain_analyses``
    (upsert — one row per unique domain, updated on repeat hits)
  • Accept user feedback submissions into ``user_feedback``
  • Manage model version registry in ``model_versions``
  • Expose curated ground-truth lookups from ``known_domains``
  • Provide admin helpers: recent-log fetch, aggregate stats, purge
  • Health-check the Supabase connection for the /health endpoint

Design principles
─────────────────
  • **Non-blocking** — ``log_analysis()`` dispatches the write to a
    ``ThreadPoolExecutor`` so the API response is never delayed by
    a Supabase round-trip.  A synchronous variant is available for
    tests and scripts.
  • **Graceful degradation** — every public method catches all
    Supabase / network errors, logs them with loguru, and returns a
    safe sentinel (``False`` / ``None`` / ``[]``).  A DB failure
    must never crash the API.
  • **Upsert, not insert** — ``domain_analyses`` keeps one row per
    domain.  Re-analyses update the existing row (FLAG 1 mitigation).
  • **Score gate** — ``log_analysis`` silently skips safe domains
    (score < ``settings.SUPABASE_LOG_THRESHOLD``), enforcing the
    blueprint rule at the source.
  • **Lazy initialisation** — the Supabase client is created on first
    use so the app starts cleanly even when credentials are absent
    (development without Supabase).

Usage
─────
    from database.supabase_client import db

    # Fire-and-forget (non-blocking)
    db.log_analysis(
        domain="paypal-secure-login.xyz",
        score=0.93,
        label="malicious",
        confidence="high",
        reasons=["Domain registered 4 days ago", "Contains phishing keywords"],
        model_version="v1.0.0",
        duration_ms=187,
        network_features_used=True,
        source="extension",
    )

    # Synchronous (tests / scripts)
    ok = db.log_analysis_sync(...)

    # Feedback
    db.insert_feedback("example.com", "malicious", "false_positive", "Looks fine to me")

    # Model version
    model = db.get_production_model()   # → ModelVersionRow | None
    db.register_model_version("v1.1.0", accuracy=0.97, ...)
    db.set_production_model("v1.1.0")

    # Admin
    rows  = db.get_recent_analyses(limit=50, label="malicious")
    stats = db.get_stats()
    db.purge_old_analyses(older_than_days=90)

    # Health
    ok = db.health_check()   # → bool
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Guard: supabase SDK import — fail loudly at import time with a clear message
# ---------------------------------------------------------------------------
try:
    from supabase import Client, create_client
    from postgrest.exceptions import APIError as PostgRESTError
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'supabase' package is not installed.  "
        "Run: pip install supabase==2.4.6"
    ) from exc

# ---------------------------------------------------------------------------
# Internal imports — config is always available (built in Phase 1)
# ---------------------------------------------------------------------------
from config.settings import settings
from config.constants import (
    AnalysisSource,
    LABEL,
    UserVerdict,
    KnownDomainVerdict,
)


# ══════════════════════════════════════════════════════════════════════════════
# Row dataclasses — typed representations of each Supabase table row
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DomainAnalysisRow:
    """Maps to the ``domain_analyses`` table."""
    id: str
    domain: str
    score: float
    label: str
    confidence: Optional[str]
    top_reasons: List[str]
    model_version: str
    analysis_duration_ms: Optional[int]
    network_features_used: bool
    source: str
    created_at: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DomainAnalysisRow":
        return cls(
            id=d.get("id", ""),
            domain=d.get("domain", ""),
            score=float(d.get("score", 0.0)),
            label=d.get("label", ""),
            confidence=d.get("confidence"),
            top_reasons=d.get("top_reasons") or [],
            model_version=d.get("model_version", ""),
            analysis_duration_ms=d.get("analysis_duration_ms"),
            network_features_used=bool(d.get("network_features_used", True)),
            source=d.get("source", "api"),
            created_at=d.get("created_at", ""),
        )


@dataclass
class FeedbackRow:
    """Maps to the ``user_feedback`` table."""
    id: str
    domain: str
    system_label: str
    user_verdict: str
    user_comment: Optional[str]
    created_at: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeedbackRow":
        return cls(
            id=d.get("id", ""),
            domain=d.get("domain", ""),
            system_label=d.get("system_label", ""),
            user_verdict=d.get("user_verdict", ""),
            user_comment=d.get("user_comment"),
            created_at=d.get("created_at", ""),
        )


@dataclass
class ModelVersionRow:
    """Maps to the ``model_versions`` table."""
    id: str
    version: str
    accuracy: Optional[float]
    f1_score: Optional[float]
    auc_roc: Optional[float]
    precision_score: Optional[float]
    recall_score: Optional[float]
    false_positive_rate: Optional[float]
    training_samples: Optional[int]
    feature_count: Optional[int]
    notes: Optional[str]
    is_production: bool
    deployed_at: Optional[str]
    created_at: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelVersionRow":
        return cls(
            id=d.get("id", ""),
            version=d.get("version", ""),
            accuracy=d.get("accuracy"),
            f1_score=d.get("f1_score"),
            auc_roc=d.get("auc_roc"),
            precision_score=d.get("precision_score"),
            recall_score=d.get("recall_score"),
            false_positive_rate=d.get("false_positive_rate"),
            training_samples=d.get("training_samples"),
            feature_count=d.get("feature_count"),
            notes=d.get("notes"),
            is_production=bool(d.get("is_production", False)),
            deployed_at=d.get("deployed_at"),
            created_at=d.get("created_at", ""),
        )


@dataclass
class KnownDomainRow:
    """Maps to the ``known_domains`` table."""
    id: str
    domain: str
    verdict: str        # 'safe' | 'malicious' | 'phishing'
    source: Optional[str]
    is_active: bool
    created_at: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KnownDomainRow":
        return cls(
            id=d.get("id", ""),
            domain=d.get("domain", ""),
            verdict=d.get("verdict", ""),
            source=d.get("source"),
            is_active=bool(d.get("is_active", True)),
            created_at=d.get("created_at", ""),
        )


@dataclass
class AnalysisStats:
    """Aggregate statistics returned by :meth:`SupabaseClient.get_stats`."""
    total_logged: int = 0
    suspicious_count: int = 0
    malicious_count: int = 0
    top_malicious_domains: List[Dict[str, Any]] = field(default_factory=list)
    latest_analysis_at: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# Main client
# ══════════════════════════════════════════════════════════════════════════════

class SupabaseClient:
    """
    Thin, fault-tolerant wrapper around the Supabase Python SDK v2.

    All public methods are safe to call even when Supabase is not configured
    or temporarily unavailable — they log a warning and return a safe default.
    """

    # Table names — single source of truth so a rename only changes one place
    TABLE_ANALYSES      = "domain_analyses"
    TABLE_FEEDBACK      = "user_feedback"
    TABLE_MODEL_VERSIONS = "model_versions"
    TABLE_KNOWN_DOMAINS  = "known_domains"

    def __init__(self) -> None:
        self._client: Optional[Client] = None
        # Single background thread for fire-and-forget log writes.
        # max_workers=2: two concurrent log writes is plenty; avoids
        # overwhelming the Supabase free tier connection pool.
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="adis-db")
        self._configured: bool = settings.supabase_configured

        if not self._configured:
            logger.warning(
                "Supabase credentials not set (SUPABASE_URL / SUPABASE_KEY). "
                "All database operations will be no-ops until credentials are provided."
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @property
    def client(self) -> Optional[Client]:
        """
        Lazy-initialise the Supabase client on first access.

        Returns ``None`` (and logs once) if credentials are missing so that
        callers can safely check ``if not self.client``.
        """
        if self._client is not None:
            return self._client

        if not self._configured:
            return None

        try:
            self._client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
            logger.info("Supabase client initialised ({})", settings.SUPABASE_URL)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialise Supabase client: {}", exc)
            self._client = None

        return self._client

    def _is_available(self) -> bool:
        """Return ``True`` when the client is ready for queries."""
        return self.client is not None

    # ══════════════════════════════════════════════════════════════════════════
    # domain_analyses — write path
    # ══════════════════════════════════════════════════════════════════════════

    def log_analysis(
        self,
        domain: str,
        score: float,
        label: str,
        confidence: str,
        reasons: List[str],
        model_version: str,
        duration_ms: Optional[int] = None,
        network_features_used: bool = True,
        source: str = AnalysisSource.EXTENSION,
    ) -> None:
        """
        Fire-and-forget: log an analysis result asynchronously.

        The write is dispatched to a background thread so the API response
        is returned to the browser extension immediately without waiting for
        the Supabase round-trip.

        Safe domains (score < ``settings.SUPABASE_LOG_THRESHOLD``) are
        silently skipped — per FLAG 1 free-tier mitigation.
        """
        # Blueprint FLAG 1: skip safe domains entirely
        if score < settings.SUPABASE_LOG_THRESHOLD:
            return

        if not self._is_available():
            return

        self._executor.submit(
            self._write_analysis,
            domain=domain,
            score=score,
            label=label,
            confidence=confidence,
            reasons=reasons,
            model_version=model_version,
            duration_ms=duration_ms,
            network_features_used=network_features_used,
            source=str(source),
        )

    def log_analysis_sync(
        self,
        domain: str,
        score: float,
        label: str,
        confidence: str,
        reasons: List[str],
        model_version: str,
        duration_ms: Optional[int] = None,
        network_features_used: bool = True,
        source: str = AnalysisSource.EXTENSION,
    ) -> bool:
        """
        Synchronous variant of ``log_analysis`` — blocks until the write
        completes or fails.  Use in tests and CLI scripts, not in request
        handlers.

        Returns ``True`` on success, ``False`` on failure or skip.
        """
        if score < settings.SUPABASE_LOG_THRESHOLD:
            logger.debug("Skipping log for safe domain '{}' (score={:.3f})", domain, score)
            return False

        if not self._is_available():
            return False

        return self._write_analysis(
            domain=domain,
            score=score,
            label=label,
            confidence=confidence,
            reasons=reasons,
            model_version=model_version,
            duration_ms=duration_ms,
            network_features_used=network_features_used,
            source=str(source),
        )

    def _write_analysis(
        self,
        domain: str,
        score: float,
        label: str,
        confidence: str,
        reasons: List[str],
        model_version: str,
        duration_ms: Optional[int],
        network_features_used: bool,
        source: str,
    ) -> bool:
        """
        Internal: perform the Supabase upsert for ``domain_analyses``.

        Uses ``upsert`` with ``on_conflict="domain"`` so re-analysed domains
        update their existing row rather than appending duplicate rows.
        This is the primary free-tier storage mitigation (FLAG 1).
        """
        payload: Dict[str, Any] = {
            "domain": domain.lower(),
            "score": round(score, 6),
            "label": label,
            "confidence": confidence,
            "top_reasons": reasons,
            "model_version": model_version,
            "analysis_duration_ms": duration_ms,
            "network_features_used": network_features_used,
            "source": source,
        }

        try:
            self.client.table(self.TABLE_ANALYSES).upsert(  # type: ignore[union-attr] 
                payload, on_conflict="domain"
            ).execute()
            logger.debug(
                "Logged analysis: domain='{}' label='{}' score={:.3f}",
                domain, label, score,
            )
            return True

        except PostgRESTError as exc:
            logger.warning(
                "Supabase PostgREST error logging analysis for '{}': {} — {}",
                domain, exc.code, exc.message,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected error logging analysis for '{}': {}", domain, exc)

        return False

    # ══════════════════════════════════════════════════════════════════════════
    # user_feedback — write path
    # ══════════════════════════════════════════════════════════════════════════

    def insert_feedback(
        self,
        domain: str,
        system_label: str,
        user_verdict: str,
        user_comment: Optional[str] = None,
    ) -> bool:
        """
        Insert a user feedback report into ``user_feedback``.

        Called by the ``POST /feedback`` route.  Returns ``True`` on success.

        ``user_verdict`` must be one of: ``'false_positive'``,
        ``'confirmed_malicious'``, ``'unsure'`` (matches Supabase CHECK).
        """
        if not self._is_available():
            return False

        valid_verdicts = {v.value for v in UserVerdict}
        if user_verdict not in valid_verdicts:
            logger.warning(
                "insert_feedback: invalid user_verdict '{}'. "
                "Must be one of {}",
                user_verdict, valid_verdicts,
            )
            return False

        payload: Dict[str, Any] = {
            "domain": domain.lower(),
            "system_label": system_label,
            "user_verdict": user_verdict,
            "user_comment": user_comment,
        }

        try:
            self.client.table(self.TABLE_FEEDBACK).insert(payload).execute()  # type: ignore[union-attr]
            logger.info(
                "Feedback recorded: domain='{}' verdict='{}'",
                domain, user_verdict,
            )
            return True

        except PostgRESTError as exc:
            logger.warning(
                "Supabase error inserting feedback for '{}': {} — {}",
                domain, exc.code, exc.message,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected error inserting feedback for '{}': {}", domain, exc)

        return False

    # ══════════════════════════════════════════════════════════════════════════
    # model_versions — read / write
    # ══════════════════════════════════════════════════════════════════════════

    def get_production_model(self) -> Optional[ModelVersionRow]:
        """
        Return the model version currently flagged ``is_production=TRUE``.

        Returns ``None`` if no model is registered or on any error.
        Used by the ``GET /version`` endpoint and at API startup.
        """
        if not self._is_available():
            return None

        try:
            result = (
                self.client.table(self.TABLE_MODEL_VERSIONS)  # type: ignore[union-attr]
                .select("*")
                .eq("is_production", True)
                .limit(1)
                .execute()
            )
            if result.data:
                return ModelVersionRow.from_dict(result.data[0])
            return None

        except Exception as exc:  # noqa: BLE001
            logger.warning("Error fetching production model version: {}", exc)
            return None
        
    def register_model_version(
        self,
        version: str,
        *,
        accuracy: Optional[float] = None,
        f1_score: Optional[float] = None,
        auc_roc: Optional[float] = None,
        precision_score: Optional[float] = None,
        recall_score: Optional[float] = None,
        false_positive_rate: Optional[float] = None,
        training_samples: Optional[int] = None,
        feature_count: Optional[int] = None,
        notes: Optional[str] = None,
        set_as_production: bool = False,
    ) -> bool:
        """
        Insert or update a model version record in ``model_versions``.

        If ``set_as_production=True``, atomically promotes this version to
        production after upserting (calls :meth:`set_production_model`).

        Called from the training pipeline after a successful training run.
        """
        if not self._is_available():
            return False

        payload: Dict[str, Any] = {
            "version": version,
            "accuracy": accuracy,
            "f1_score": f1_score,
            "auc_roc": auc_roc,
            "precision_score": precision_score,
            "recall_score": recall_score,
            "false_positive_rate": false_positive_rate,
            "training_samples": training_samples,
            "feature_count": feature_count,
            "notes": notes,
        }
        # Strip None values so existing columns aren't overwritten with NULL
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            self.client.table(self.TABLE_MODEL_VERSIONS).upsert(  # type: ignore[union-attr]
                payload, on_conflict="version"
            ).execute()
            logger.info("Registered model version '{}'", version)

            if set_as_production:
                return self.set_production_model(version)

            return True

        except PostgRESTError as exc:
            logger.warning(
                "Supabase error registering model version '{}': {} — {}",
                version, exc.code, exc.message,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected error registering model version '{}': {}", version, exc)

        return False

    def set_production_model(self, version: str) -> bool:
        """
        Atomically promote ``version`` to production.

        Steps:
          1. Clear ``is_production=TRUE`` on all rows (Supabase's partial
             unique index enforces only one TRUE at a time, but we must
             demote the current prod row first to avoid a conflict).
          2. Set ``is_production=TRUE`` on the target version, and record
             ``deployed_at`` as the current UTC timestamp.

        Returns ``True`` if both operations succeed.
        """
        if not self._is_available():
            return False

        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            # Step 1 — demote all current production rows
            self.client.table(self.TABLE_MODEL_VERSIONS).update(  # type: ignore[union-attr]
                {"is_production": False}
            ).eq("is_production", True).execute()

            # Step 2 — promote target version
            self.client.table(self.TABLE_MODEL_VERSIONS).update(  # type: ignore[union-attr]
                {"is_production": True, "deployed_at": now_iso}
            ).eq("version", version).execute()

            logger.info("Set production model to '{}'", version)
            return True

        except PostgRESTError as exc:
            logger.error(
                "Supabase error setting production model to '{}': {} — {}",
                version, exc.code, exc.message,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Unexpected error setting production model to '{}': {}", version, exc,
            )

        return False

    # ══════════════════════════════════════════════════════════════════════════
    # known_domains — curated ground-truth lookups
    # ══════════════════════════════════════════════════════════════════════════
    
    #Manual edit that strips www. and subdomains for known domain lookups

    @staticmethod
    def _match_key(domain: str) -> str:
        """
    Normalize a domain to the form stored in known_domains for MATCHING only.

    The known_domains table stores bare registrable domains (e.g. 'wikipedia.org'),
    but the analyze pipeline deliberately preserves a leading 'www.' (the model's
    has_www / num_subdomains features depend on it). So we strip a single leading
    'www.' here, at lookup time only — the caller's domain, features, cache key,
    and logged value are all unaffected.
    """
        d = domain.lower()
        return d[4:] if d.startswith("www.") else d

    def lookup_known_domain(self, domain: str) -> Optional[KnownDomainRow]:
        """
        Check whether ``domain`` exists in the curated ``known_domains``
        table with ``is_active=TRUE``.

        Returns a :class:`KnownDomainRow` if found, ``None`` otherwise.

        The analyze route uses this to short-circuit ML inference for domains
        whose ground truth is already known (e.g. from PhishTank or manual
        curation), returning a deterministic result instantly.
        """
        if not self._is_available():
            return None

        try:
            result = (
                self.client.table(self.TABLE_KNOWN_DOMAINS)  # type: ignore[union-attr]
                .select("*")
                .eq("domain", self._match_key(domain))
                .eq("is_active", True)
                .limit(1)
                .execute()
            )
            if result.data:
                return KnownDomainRow.from_dict(result.data[0])
            return None

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Error looking up known domain '{}': {}", domain, exc,
            )
            return None

    def upsert_known_domain(
        self,
        domain: str,
        verdict: str,
        source: Optional[str] = None,
        is_active: bool = True,
    ) -> bool:
        """
        Insert or update a record in ``known_domains``.

        ``verdict`` must be one of ``'safe'``, ``'malicious'``,
        ``'phishing'`` (matches the Supabase CHECK constraint).

        Used by the threat-feed ingestion scripts (PhishTank, OpenPhish,
        URLhaus, Tranco) to populate the curated list.
        """
        if not self._is_available():
            return False

        valid_verdicts = {v.value for v in KnownDomainVerdict}
        if verdict not in valid_verdicts:
            logger.warning(
                "upsert_known_domain: invalid verdict '{}'. Must be one of {}",
                verdict, valid_verdicts,
            )
            return False

        payload: Dict[str, Any] = {
            "domain": domain.lower(),
            "verdict": verdict,
            "source": source,
            "is_active": is_active,
        }

        try:
            self.client.table(self.TABLE_KNOWN_DOMAINS).upsert(  # type: ignore[union-attr]
                payload, on_conflict="domain"
            ).execute()
            logger.debug(
                "Known domain upserted: '{}' verdict='{}' source='{}'",
                domain, verdict, source,
            )
            return True

        except PostgRESTError as exc:
            logger.warning(
                "Supabase error upserting known domain '{}': {} — {}",
                domain, exc.code, exc.message,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Unexpected error upserting known domain '{}': {}", domain, exc,
            )

        return False

    # ══════════════════════════════════════════════════════════════════════════
    # Admin / reporting
    # ══════════════════════════════════════════════════════════════════════════

    def get_recent_analyses(
        self,
        limit: int = 50,
        label: Optional[str] = None,
    ) -> List[DomainAnalysisRow]:
        """
        Fetch the most-recently logged domain analyses, newest first.

        Used by ``GET /admin/logs``.

        Args:
            limit: Maximum number of rows to return (capped at 500).
            label: Optional filter — ``'suspicious'`` or ``'malicious'``.
                   ``None`` returns both.
        """
        if not self._is_available():
            return []

        limit = min(limit, 500)

        try:
            query = (
                self.client.table(self.TABLE_ANALYSES)  # type: ignore[union-attr]
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
            )

            if label is not None:
                valid_log_labels = {LABEL.SUSPICIOUS, LABEL.MALICIOUS}
                if label not in valid_log_labels:
                    logger.warning(
                        "get_recent_analyses: invalid label '{}'. "
                        "Must be one of {}",
                        label, valid_log_labels,
                    )
                    return []
                query = query.eq("label", label)

            result = query.execute()
            return [DomainAnalysisRow.from_dict(row) for row in (result.data or [])]

        except Exception as exc:  # noqa: BLE001
            logger.warning("Error fetching recent analyses: {}", exc)
            return []

    def get_stats(self) -> AnalysisStats:
        """
        Return aggregate statistics from ``domain_analyses``.

        Used by ``GET /stats`` (authenticated).  Supabase RPC or
        client-side aggregation over a capped result set.

        Note: For very large tables, replace this with a Supabase
        database function (``CREATE FUNCTION adis_stats()``) to keep
        the query fast.
        """
        if not self._is_available():
            return AnalysisStats()

        try:
            # Total + per-label counts via two cheap queries
            all_result = (
                self.client.table(self.TABLE_ANALYSES)  # type: ignore[union-attr]
                .select("id, label, created_at", count="exact")
                .execute()
            )

            rows = all_result.data or []
            total = all_result.count or len(rows)

            suspicious_count = sum(1 for r in rows if r.get("label") == LABEL.SUSPICIOUS)
            malicious_count  = sum(1 for r in rows if r.get("label") == LABEL.MALICIOUS)

            # Latest timestamp
            latest = rows[0].get("created_at") if rows else None

            # Top 10 most-logged malicious domains (simple client-side tally)
            domain_counts: Dict[str, int] = {}
            for row in rows:
                if row.get("label") == LABEL.MALICIOUS:
                    d = row.get("domain", "")
                    domain_counts[d] = domain_counts.get(d, 0) + 1

            top_malicious = sorted(
                [{"domain": k, "count": v} for k, v in domain_counts.items()],
                key=lambda x: x["count"],
                reverse=True,
            )[:10]

            return AnalysisStats(
                total_logged=total,
                suspicious_count=suspicious_count,
                malicious_count=malicious_count,
                top_malicious_domains=top_malicious,
                latest_analysis_at=latest,
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning("Error fetching stats: {}", exc)
            return AnalysisStats()

    def purge_old_analyses(self, older_than_days: int = 90) -> int:
        """
        Delete ``domain_analyses`` rows older than ``older_than_days`` days.

        Intended for the monthly cleanup job referenced in FLAG 1.
        Returns the number of rows deleted, or -1 on error.

        Example cron invocation::

            from database.supabase_client import db
            deleted = db.purge_old_analyses(older_than_days=90)
            print(f"Purged {deleted} old rows")
        """
        if not self._is_available():
            return -1

        try:
            # Build an ISO-8601 cutoff timestamp
            from datetime import timedelta
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=older_than_days)
            ).isoformat()

            result = (
                self.client.table(self.TABLE_ANALYSES)  # type: ignore[union-attr]
                .delete(count="exact")
                .lt("created_at", cutoff)
                .execute()
            )

            deleted = result.count or 0
            logger.info(
                "Purged {} analyses older than {} days (cutoff: {})",
                deleted, older_than_days, cutoff,
            )
            return deleted

        except PostgRESTError as exc:
            logger.error(
                "Supabase error during purge: {} — {}", exc.code, exc.message,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error during purge: {}", exc)

        return -1

    # ══════════════════════════════════════════════════════════════════════════
    # Health check
    # ══════════════════════════════════════════════════════════════════════════

    def health_check(self) -> bool:
        """
        Ping Supabase with a lightweight query to confirm connectivity.

        Used by ``GET /health`` to include database status.  Returns
        ``True`` if Supabase responds successfully, ``False`` otherwise.
        """
        if not self._is_available():
            return False

        try:
            # A single-row, single-column select is the cheapest possible ping
            self.client.table(self.TABLE_ANALYSES).select("id").limit(1).execute()  # type: ignore[union-attr]
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Supabase health check failed: {}", exc)
            return False

    # ══════════════════════════════════════════════════════════════════════════
    # Resource management
    # ══════════════════════════════════════════════════════════════════════════

    def shutdown(self, wait: bool = True) -> None:
        """
        Gracefully shut down the background thread pool.

        Call this on application exit (Gunicorn worker teardown) to flush
        any pending fire-and-forget log writes before the process exits.

        Args:
            wait: If ``True`` (default), block until all queued writes
                  complete.  Set ``False`` for a fast but possibly lossy
                  shutdown.
        """
        logger.debug("Shutting down Supabase client executor (wait={})", wait)
        self._executor.shutdown(wait=wait)

    def __repr__(self) -> str:
        url = settings.SUPABASE_URL or "(not configured)"
        return (
            f"<SupabaseClient url={url!r} "
            f"configured={self._configured} "
            f"client_ready={self._client is not None}>"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Singleton — import this everywhere
# ══════════════════════════════════════════════════════════════════════════════

db = SupabaseClient()
