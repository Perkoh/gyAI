"""
features/network.py
====================
gyAI — AI-Powered Domain Intelligence System(ADIS)

Network feature extraction: features 31–48 of the 48-feature vector.

These features require **live** DNS and WHOIS lookups, so — unlike the purely
lexical features in ``structural.py`` — they can be slow, rate-limited, or fail
outright. Per FLAG 4 of the project blueprint every network call therefore:

* runs with a hard timeout (DNS ~2s per query, WHOIS 3s max),
* is wrapped in try/except so a single failed lookup never aborts the analysis,
* falls back to a documented default value on failure, and
* is reflected in the meta flag ``network_features_available`` so the API can
  tell the caller whether the network layer contributed real data.

The public entry point is :func:`extract_network_features`, which returns a
plain ``dict`` keyed by the exact feature names from blueprint §6.2. The
:mod:`features.assembler` module is responsible for ordering these values into
the numpy vector using ``features/constants.py``; this module only extracts.

Design notes
------------
* DNS record-type queries (A, AAAA, MX, NS, TXT) are issued **concurrently** so
  the worst-case DNS wall-clock stays close to a single query timeout rather
  than the sum of all five.
* The DNS block and the WHOIS block also run concurrently with respect to each
  other, keeping total network time at roughly ``max(dns, whois)`` (~3s) — the
  target from the Phase 2 benchmark ("network features < 3s worst case").
* Caching of these lookups (Redis, 24h TTL — FLAG 4) is intentionally handled by
  the calling layer (the analyze route / assembler), not here, so this module
  stays a pure, side-effect-free extractor that is trivial to unit-test offline.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# --- Third-party (see requirements.txt) --------------------------------------
import dns.resolver
import dns.exception
import tldextract
import whois  # python-whois

# --- Logging (loguru per blueprint §8.1, with a stdlib fallback) -------------
try:  # pragma: no cover - trivial import shim
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("adis.features.network")

# --- Optional central config (config/settings.py). Fall back to sane defaults
# so the extractor also works standalone (e.g. inside the training pipeline).
try:  # pragma: no cover - config is environment-specific
    from config import settings as _settings  # type: ignore

    DNS_TIMEOUT: float = float(getattr(_settings, "DNS_TIMEOUT", 2.0))
    WHOIS_TIMEOUT: float = float(getattr(_settings, "WHOIS_TIMEOUT", 3.0))
except Exception:  # pragma: no cover
    DNS_TIMEOUT = 2.0
    WHOIS_TIMEOUT = 3.0


# =============================================================================
# Constants
# =============================================================================

#: Threshold (days) below which a domain is considered "newly registered".
NEWLY_REGISTERED_THRESHOLD_DAYS = 30

#: Fast-flux heuristic thresholds (blueprint feature 45).
FAST_FLUX_MAX_TTL = 300
FAST_FLUX_MIN_A_RECORDS = 3

#: The 18 network feature names, in blueprint §6.2 order (31 → 48).
NETWORK_FEATURE_NAMES: List[str] = [
    "domain_age_days",             # 31
    "is_newly_registered",         # 32
    "days_until_expiry",           # 33
    "registration_length_days",    # 34
    "registrar_is_common",         # 35
    "whois_country",               # 36
    "whois_privacy_enabled",       # 37
    "has_a_record",                # 38
    "num_a_records",               # 39
    "has_mx_record",               # 40
    "has_ns_record",               # 41
    "num_ns_records",              # 42
    "has_txt_record",              # 43
    "dns_ttl",                     # 44
    "is_fast_flux",                # 45
    "has_ipv6",                    # 46
    "dns_resolves",                # 47
    "network_features_available",  # 48
]

#: Default value for every network feature when the corresponding lookup is
#: unavailable (exact defaults from the blueprint §6.2 table).
NETWORK_FEATURE_DEFAULTS: Dict[str, Any] = {
    "domain_age_days": -1,
    "is_newly_registered": False,
    "days_until_expiry": -1,
    "registration_length_days": -1,
    "registrar_is_common": False,
    "whois_country": "unknown",
    "whois_privacy_enabled": False,
    "has_a_record": False,
    "num_a_records": 0,
    "has_mx_record": False,
    "has_ns_record": False,
    "num_ns_records": 0,
    "has_txt_record": False,
    "dns_ttl": -1,
    "is_fast_flux": False,
    "has_ipv6": False,
    "dns_resolves": False,
    "network_features_available": False,
}

#: Substrings that identify well-known / mainstream registrars. Used by
#: ``registrar_is_common`` (feature 35). Matched case-insensitively against the
#: WHOIS ``registrar`` field. Kept here (rather than config/constants.py, which
#: holds the lexical keyword/brand/TLD lists) because it is network-specific.
COMMON_REGISTRARS: frozenset = frozenset(
    {
        "godaddy",
        "namecheap",
        "google",
        "cloudflare",
        "tucows",
        "network solutions",
        "name.com",
        "gandi",
        "hover",
        "porkbun",
        "dynadot",
        "enom",
        "ionos",
        "1&1",
        "ovh",
        "hostgator",
        "bluehost",
        "squarespace",
        "wix",
        "amazon",
        "markmonitor",
        "csc corporate",
        "register.com",
        "domain.com",
        "namesilo",
        "netim",
        "epik",
        "key-systems",
        "wild west domains",
    }
)

#: Substrings that indicate the WHOIS record is behind a privacy/proxy service
#: or has been GDPR-redacted. Used by ``whois_privacy_enabled`` (feature 37).
PRIVACY_KEYWORDS: tuple = (
    "redacted",
    "privacy",
    "whoisguard",
    "private",
    "protected",
    "withheld",
    "data protected",
    "domains by proxy",
    "perfect privacy",
    "contact privacy",
    "identity protection",
    "gdpr",
    "not disclosed",
    "statutory masking",
    "proxy",
    "anonymize",
)


# =============================================================================
# Small helpers
# =============================================================================

def _normalize_domain(domain: str) -> str:
    """Lower-case and strip scheme/path/whitespace from a raw domain string."""
    d = (domain or "").strip().lower()
    if "://" in d:
        d = d.split("://", 1)[1]
    # Drop any path, query, port, or userinfo — we only want the host.
    d = d.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    d = d.split("@")[-1]
    d = d.split(":", 1)[0]
    return d.strip(".")


def _registered_domain(hostname: str) -> str:
    """Return the registrable domain (SLD+TLD) for WHOIS lookups.

    WHOIS records exist for the registrable domain, never for arbitrary
    subdomains, so ``a.b.example.co.uk`` must be reduced to ``example.co.uk``.
    """
    ext = tldextract.extract(hostname)
    if ext.registered_domain:
        return ext.registered_domain
    return hostname  # fall back to whatever we were given


def _as_single(value: Any) -> Any:
    """WHOIS fields are frequently lists; collapse to the first element.

    For dates we take the *earliest* creation and *latest* expiration is handled
    by the callers; here we just unwrap single-element ambiguity safely.
    """
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _to_naive_utc(dt: Any) -> Optional[datetime]:
    """Coerce a WHOIS datetime (possibly tz-aware or a list) to naive UTC."""
    dt = _as_single(dt)
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _run_with_timeout(func, timeout: float, label: str):
    """Run ``func()`` in a worker thread, returning ``None`` on timeout/error.

    ``python-whois`` exposes no timeout parameter of its own, so we enforce one
    externally. A timed-out thread may keep running in the background until its
    own socket times out, but the caller is never blocked past ``timeout``.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeout:
        logger.warning(f"[network] {label} timed out after {timeout:.1f}s")
        return None
    except Exception as exc:  # noqa: BLE001 - any lookup failure is non-fatal
        logger.debug(f"[network] {label} failed: {exc!r}")
        return None
    finally:
        # Do not wait on a possibly-hung lookup thread.
        executor.shutdown(wait=False)


# =============================================================================
# DNS extraction (features 38–47)
# =============================================================================

def _build_resolver(timeout: float) -> dns.resolver.Resolver:
    resolver = dns.resolver.Resolver()
    resolver.timeout = timeout   # per-server attempt
    resolver.lifetime = timeout  # total time budget for one query
    return resolver


def _query(resolver: dns.resolver.Resolver, domain: str, rtype: str):
    """Resolve a single record type. Returns the dnspython answer or ``None``.

    Every "domain has no such record" outcome (NXDOMAIN, NoAnswer, ...) as well
    as timeouts are swallowed and reported as ``None`` — the absence of a record
    is itself a valid, meaningful feature value, not an error.
    """
    try:
        return resolver.resolve(domain, rtype)
    except (
        dns.resolver.NoAnswer,
        dns.resolver.NXDOMAIN,
        dns.resolver.NoNameservers,
        dns.resolver.LifetimeTimeout,
        dns.exception.Timeout,
    ):
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[network] DNS {rtype} query for {domain} failed: {exc!r}")
        return None


def _extract_dns_features(hostname: str, timeout: float) -> Dict[str, Any]:
    """Compute the ten DNS features (38–47) for ``hostname``.

    Issues the five record-type queries concurrently to keep total DNS latency
    near a single query timeout rather than the sum of all queries.
    """
    features: Dict[str, Any] = {
        k: NETWORK_FEATURE_DEFAULTS[k]
        for k in (
            "has_a_record", "num_a_records", "has_mx_record",
            "has_ns_record", "num_ns_records", "has_txt_record",
            "dns_ttl", "is_fast_flux", "has_ipv6", "dns_resolves",
        )
    }

    resolver = _build_resolver(timeout)
    record_types = ("A", "AAAA", "MX", "NS", "TXT")

    answers: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=len(record_types)) as pool:
        futures = {
            rtype: pool.submit(_query, resolver, hostname, rtype)
            for rtype in record_types
        }
        for rtype, fut in futures.items():
            try:
                answers[rtype] = fut.result(timeout=timeout + 0.5)
            except Exception:  # noqa: BLE001
                answers[rtype] = None

    a_ans = answers.get("A")
    if a_ans is not None:
        features["has_a_record"] = True
        features["num_a_records"] = len(a_ans)
        try:
            features["dns_ttl"] = int(a_ans.rrset.ttl)
        except Exception:  # noqa: BLE001
            features["dns_ttl"] = -1

    aaaa_ans = answers.get("AAAA")
    if aaaa_ans is not None and len(aaaa_ans) > 0:
        features["has_ipv6"] = True

    mx_ans = answers.get("MX")
    if mx_ans is not None and len(mx_ans) > 0:
        features["has_mx_record"] = True

    ns_ans = answers.get("NS")
    if ns_ans is not None:
        features["has_ns_record"] = len(ns_ans) > 0
        features["num_ns_records"] = len(ns_ans)

    txt_ans = answers.get("TXT")
    if txt_ans is not None and len(txt_ans) > 0:
        features["has_txt_record"] = True

    # dns_resolves: did *any* query return an answer at all?
    features["dns_resolves"] = any(ans is not None for ans in answers.values())

    # is_fast_flux (feature 45): low TTL AND many A records — classic fast-flux.
    features["is_fast_flux"] = (
        0 <= features["dns_ttl"] < FAST_FLUX_MAX_TTL
        and features["num_a_records"] > FAST_FLUX_MIN_A_RECORDS
    )

    return features


# =============================================================================
# WHOIS extraction (features 31–37)
# =============================================================================

def _whois_privacy_enabled(record: Any) -> bool:
    """Detect privacy/proxy/GDPR redaction anywhere in the WHOIS record."""
    haystacks: List[str] = []

    raw = getattr(record, "text", None)
    if isinstance(raw, str):
        haystacks.append(raw)

    for field in ("registrar", "org", "name", "emails", "registrant"):
        val = getattr(record, field, None) if not isinstance(record, dict) else record.get(field)
        if isinstance(val, (list, tuple)):
            haystacks.extend(str(v) for v in val)
        elif val is not None:
            haystacks.append(str(val))

    blob = " ".join(haystacks).lower()
    return any(keyword in blob for keyword in PRIVACY_KEYWORDS)


def _extract_whois_features(hostname: str, timeout: float) -> Dict[str, Any]:
    """Compute the seven WHOIS features (31–37) for ``hostname``.

    Returns defaults for any field that cannot be resolved. The whole WHOIS
    call is bounded by ``timeout`` (FLAG 4) because python-whois has no native
    timeout of its own.
    """
    features: Dict[str, Any] = {
        k: NETWORK_FEATURE_DEFAULTS[k]
        for k in (
            "domain_age_days", "is_newly_registered", "days_until_expiry",
            "registration_length_days", "registrar_is_common",
            "whois_country", "whois_privacy_enabled",
        )
    }

    registrable = _registered_domain(hostname)
    record = _run_with_timeout(
        lambda: whois.whois(registrable), timeout, label=f"WHOIS({registrable})"
    )
    if record is None:
        return features

    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # --- Dates (features 31, 32, 33, 34) -------------------------------------
    creation = _to_naive_utc(getattr(record, "creation_date", None))
    expiration = _to_naive_utc(getattr(record, "expiration_date", None))

    if creation is not None:
        age = (now - creation).days
        if age >= 0:
            features["domain_age_days"] = age
            features["is_newly_registered"] = age < NEWLY_REGISTERED_THRESHOLD_DAYS

    if expiration is not None:
        features["days_until_expiry"] = (expiration - now).days

    if creation is not None and expiration is not None:
        reg_len = (expiration - creation).days
        if reg_len >= 0:
            features["registration_length_days"] = reg_len

    # --- Registrar (feature 35) ----------------------------------------------
    registrar = _as_single(getattr(record, "registrar", None))
    if isinstance(registrar, str) and registrar.strip():
        reg_lower = registrar.lower()
        features["registrar_is_common"] = any(
            known in reg_lower for known in COMMON_REGISTRARS
        )

    # --- Country (feature 36) — returned raw; encoded downstream --------------
    country = _as_single(getattr(record, "country", None))
    if isinstance(country, str) and country.strip():
        features["whois_country"] = country.strip().lower()

    # --- Privacy (feature 37) ------------------------------------------------
    features["whois_privacy_enabled"] = _whois_privacy_enabled(record)

    return features


# =============================================================================
# Public API
# =============================================================================

def extract_network_features(
    domain: str,
    *,
    dns_timeout: float = DNS_TIMEOUT,
    whois_timeout: float = WHOIS_TIMEOUT,
) -> Dict[str, Any]:
    """Extract all 18 network features (31–48) for ``domain``.

    The DNS block and the WHOIS block are executed concurrently so total
    wall-clock time is roughly ``max(dns_timeout, whois_timeout)`` in the worst
    case rather than their sum.

    Parameters
    ----------
    domain:
        Raw domain string. Any scheme, path, port, or userinfo is stripped.
    dns_timeout:
        Per-query DNS time budget in seconds (default from config / 2.0s).
    whois_timeout:
        Total WHOIS time budget in seconds (default from config / 3.0s).

    Returns
    -------
    dict
        A dict keyed by every name in :data:`NETWORK_FEATURE_NAMES`. On total
        failure the blueprint defaults are returned with
        ``network_features_available = False``. Values are returned in their
        natural Python types (bools, ints, the raw ``whois_country`` string);
        categorical encoding is the assembler's/preprocessor's responsibility.
    """
    features: Dict[str, Any] = dict(NETWORK_FEATURE_DEFAULTS)

    hostname = _normalize_domain(domain)
    if not hostname:
        logger.warning(f"[network] empty/invalid domain input: {domain!r}")
        return features

    # Run DNS and WHOIS concurrently.
    dns_features: Dict[str, Any] = {}
    whois_features: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        dns_future = pool.submit(_extract_dns_features, hostname, dns_timeout)
        whois_future = pool.submit(_extract_whois_features, hostname, whois_timeout)

        # Generous hard caps; the inner calls already enforce their own budgets.
        try:
            dns_features = dns_future.result(timeout=dns_timeout + 2.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[network] DNS extraction failed for {hostname}: {exc!r}")
        try:
            whois_features = whois_future.result(timeout=whois_timeout + 2.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[network] WHOIS extraction failed for {hostname}: {exc!r}")

    features.update(dns_features)
    features.update(whois_features)

    # Meta flag (feature 48): did the network layer contribute any real data?
    dns_ok = bool(dns_features.get("dns_resolves", False))
    whois_ok = features.get("domain_age_days", -1) != -1 or features.get(
        "whois_country", "unknown"
    ) != "unknown"
    features["network_features_available"] = dns_ok or whois_ok

    return features


def default_network_features() -> Dict[str, Any]:
    """Return a fresh copy of the all-defaults network feature dict.

    Useful for the training pipeline / assembler when network lookups are
    disabled or being computed elsewhere.
    """
    return dict(NETWORK_FEATURE_DEFAULTS)


__all__ = [
    "extract_network_features",
    "default_network_features",
    "NETWORK_FEATURE_NAMES",
    "NETWORK_FEATURE_DEFAULTS",
]