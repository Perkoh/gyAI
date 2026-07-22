"""
gyAI — AI-Powered Domain Intelligence System (ADIS)
ml/reason_mapper.py

Maps model features to human-readable reason strings.

Per PROJECT_BLUEPRINT.md section 6.4, after the LightGBM model scores a domain,
SHAP tells us *which* features pushed the prediction toward "malicious" and by
how much. This module turns those features into the plain-English sentences the
browser extension shows the user (blueprint sections 6.4, 10.2, 11.2).

This module is pure Python — no numpy, no model, no SHAP — so it is trivially
unit-testable and reused identically at every layer. `ml/explainer.py` computes
the SHAP ranking and calls `build_reasons()` here to render the final strings.

Design notes
------------
* Each feature maps to a handler that returns a `Reason` (text + category) or
  None. Returning None means "no sensible reason for this feature/value", so the
  explainer moves on to the next-ranked feature. This double-gate (SHAP says the
  feature mattered AND the value supports a human explanation) prevents nonsense
  reasons like "registered -1 days ago" when WHOIS was unavailable.
* Handlers are gated on the *value* so a reason only fires in its suspicious
  direction (e.g. has_mx_record only explains when the record is absent).
* Reasons carry a `category` so `build_reasons()` can de-duplicate: several
  features describe the same underlying signal (domain_age_days and
  is_newly_registered are both "domain age"), and the user should see it once.
* Richer, specific phrasings (the actual keyword, TLD, brand, or look-alike
  domain) are used when the caller passes a `context` dict; otherwise a generic
  fallback is used. The feature-extraction pipeline already computes those
  values, so it can pass them through for the best wording.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Canonical feature order (blueprint sections 6.1 & 6.2, features #1..#48).
# features/constants.py is authoritative once built; this mirrors it so the
# module is self-contained and testable in isolation.
# --------------------------------------------------------------------------- #

FEATURE_NAMES: List[str] = [
    # --- structural / lexical (1..30) ---
    "domain_length", "sld_length", "tld", "tld_is_suspicious", "num_dots",
    "num_hyphens", "num_digits", "digit_ratio", "entropy", "vowel_ratio",
    "consonant_ratio", "longest_consonant_run", "num_unique_chars",
    "char_repeat_ratio", "has_ip_in_name", "is_punycode", "num_subdomains",
    "has_www", "sld_is_numeric", "contains_phishing_keyword",
    "num_phishing_keywords", "contains_brand_name", "brand_in_subdomain",
    "typosquat_distance", "is_typosquat_candidate", "hex_ratio",
    "num_special_chars", "subdomain_entropy", "tld_length",
    "ratio_digits_to_letters",
    # --- network: DNS + WHOIS (31..48) ---
    "domain_age_days", "is_newly_registered", "days_until_expiry",
    "registration_length_days", "registrar_is_common", "whois_country",
    "whois_privacy_enabled", "has_a_record", "num_a_records", "has_mx_record",
    "has_ns_record", "num_ns_records", "has_txt_record", "dns_ttl",
    "is_fast_flux", "has_ipv6", "dns_resolves", "network_features_available",
]

# Features that come from live DNS/WHOIS. When network lookups failed
# (network_features_available is falsy) these carry default values and must not
# be turned into reasons.
NETWORK_FEATURES: frozenset[str] = frozenset({
    "domain_age_days", "is_newly_registered", "days_until_expiry",
    "registration_length_days", "registrar_is_common", "whois_country",
    "whois_privacy_enabled", "has_a_record", "num_a_records", "has_mx_record",
    "has_ns_record", "num_ns_records", "has_txt_record", "dns_ttl",
    "is_fast_flux", "has_ipv6", "dns_resolves", "network_features_available",
})


@dataclass(frozen=True)
class Reason:
    """A rendered reason plus metadata used for de-duplication and debugging."""
    feature: str
    category: str
    text: str


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _as_int(value: Any) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _truthy(value: Any) -> bool:
    # Booleans arrive as 0.0/1.0 from a numpy feature vector.
    try:
        return float(value) >= 0.5
    except (TypeError, ValueError):
        return bool(value)


def _quote_list(items: Sequence[str]) -> str:
    return ", ".join(f"'{i}'" for i in items)


def _phishing_keyword_text(value: Any, ctx: Dict[str, Any]) -> Optional[str]:
    """Shared renderer for the two phishing-keyword features."""
    kws = ctx.get("matched_keywords") or ctx.get("phishing_keywords")
    if kws:
        kws = [str(k) for k in kws if str(k)]
    if kws:
        if len(kws) == 1:
            return f"The domain name contains the word {_quote_list(kws)}"
        return f"The domain name contains phishing keywords: {_quote_list(kws)}"
    n = _as_int(value)
    if n >= 2:
        return "The domain name contains several phishing-related keywords"
    return "The domain name contains a phishing-related keyword"


# --------------------------------------------------------------------------- #
# Per-feature handlers: (value, ctx) -> Optional[str]
# Each returns the reason text, or None when no reason applies for that value.
# --------------------------------------------------------------------------- #

def _h_domain_age_days(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    days = _as_int(v)
    if 0 <= days <= 60:
        if days <= 7:
            return f"This domain was registered only {days} days ago"
        return f"This domain was registered {days} days ago"
    return None


def _h_is_newly_registered(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _truthy(v):
        return "Registered less than 30 days ago — a high-risk window for phishing"
    return None


def _h_days_until_expiry(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    days = _as_int(v)
    if 0 <= days <= 60:
        return f"The domain is set to expire in {days} days, typical of throwaway sites"
    return None


def _h_registration_length(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    days = _as_int(v)
    if 0 <= days <= 400:
        return "The domain was registered for only a short period, common for phishing"
    return None


def _h_registrar_is_common(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if not _truthy(v):
        return "The domain uses a registrar frequently associated with abuse"
    return None


def _h_whois_privacy(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _truthy(v):
        return "The domain's registration details are hidden from public records"
    return None


def _h_has_a_record(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if not _truthy(v):
        return "The domain does not resolve to any server (no DNS A record)"
    return None


def _h_num_a_records(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    n = _as_int(v)
    if n >= 4:
        return f"The domain rotates across many IP addresses ({n}), used to dodge blocklists"
    return None


def _h_has_mx_record(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if not _truthy(v):
        return "No email server is configured, which is unusual for a legitimate site"
    return None


def _h_has_ns_record(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if not _truthy(v):
        return "The domain is missing standard name-server (NS) records"
    return None


def _h_has_txt_record(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if not _truthy(v):
        return "The domain has no email-authentication records (SPF/DMARC)"
    return None


def _h_dns_ttl(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    ttl = _as_int(v)
    if 0 <= ttl < 300:
        return "The domain's DNS is configured for rapid IP rotation (fast-flux)"
    return None


def _h_is_fast_flux(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _truthy(v):
        return "The domain is using fast-flux infrastructure to evade detection"
    return None


def _h_dns_resolves(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if not _truthy(v):
        return "The domain does not resolve at all, suggesting it is brand-new or abandoned"
    return None


def _h_tld_is_suspicious(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _truthy(v):
        tld = ctx.get("tld")
        if tld:
            tld = str(tld)
            if not tld.startswith("."):
                tld = "." + tld
            return f"The domain uses a high-risk extension ({tld})"
        return "The domain uses a high-risk top-level domain"
    return None


def _h_contains_phishing_keyword(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _truthy(v):
        return _phishing_keyword_text(ctx.get("num_phishing_keywords", 1), ctx)
    return None


def _h_num_phishing_keywords(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _as_int(v) >= 1:
        return _phishing_keyword_text(v, ctx)
    return None


def _h_contains_brand_name(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _truthy(v):
        brand = ctx.get("brand")
        if brand:
            return f"The domain name imitates the brand '{brand}'"
        return "The domain name imitates a well-known brand"
    return None


def _h_brand_in_subdomain(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _truthy(v):
        brand = ctx.get("brand")
        if brand:
            return f"'{brand}' appears in a subdomain, not the main domain"
        return "A well-known brand appears in a subdomain rather than the real domain"
    return None


def _h_is_typosquat(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _truthy(v):
        sim = ctx.get("similar_domain")
        if sim:
            return f"This looks like a misspelling of '{sim}'"
        return "This looks like a misspelling of a popular website"
    return None


def _h_typosquat_distance(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    d = _as_int(v)
    if 1 <= d <= 2:
        return _h_is_typosquat(1, ctx)
    return None


def _h_num_hyphens(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _as_int(v) >= 2:
        return "Excessive hyphens are used to mimic a legitimate brand name"
    return None


def _h_entropy(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _as_float(v) >= 3.5:
        return "The domain name appears randomly generated"
    return None


def _h_longest_consonant_run(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _as_int(v) >= 5:
        return "The domain name has a long run of consonants, suggesting it is auto-generated"
    return None


def _h_char_repeat_ratio(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _as_float(v) >= 0.4:
        return "The domain name repeats characters in a pattern typical of auto-generated names"
    return None


def _h_hex_ratio(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _as_float(v) >= 0.8:
        return "The domain name looks like a random hexadecimal string"
    return None


def _h_subdomain_entropy(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _as_float(v) >= 3.0:
        return "The subdomain portion appears randomly generated"
    return None


def _h_num_digits(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    n = _as_int(v)
    if n >= 4:
        return f"The domain name contains an unusual number of digits ({n})"
    return None


def _h_digit_ratio(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _as_float(v) >= 0.35:
        return "A large share of the domain name is digits, common in auto-generated domains"
    return None


def _h_ratio_digits_to_letters(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _as_float(v) >= 0.5:
        return "A large share of the domain name is digits, common in auto-generated domains"
    return None


def _h_has_ip_in_name(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _truthy(v):
        return "The domain is disguised to look like an IP address"
    return None


def _h_is_punycode(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _truthy(v):
        return "The domain uses look-alike characters that can imitate a real site (possible homograph attack)"
    return None


def _h_sld_is_numeric(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _truthy(v):
        return "The main domain name is entirely numeric, which is rare for legitimate sites"
    return None


def _h_num_subdomains(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    n = _as_int(v)
    if n >= 3:
        return f"The site is buried under multiple subdomains ({n} levels)"
    return None


def _h_num_dots(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _as_int(v) >= 4:
        return "The domain has an unusual number of sub-levels"
    return None


def _h_domain_length(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    n = _as_int(v)
    if n >= 30:
        return f"The domain name is unusually long ({n} characters)"
    return None


def _h_sld_length(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _as_int(v) >= 20:
        return "The main part of the domain name is unusually long"
    return None


def _h_num_special_chars(v: Any, ctx: Dict[str, Any]) -> Optional[str]:
    if _as_int(v) >= 1:
        return "The domain contains unusual special characters"
    return None


# Feature -> (category, handler). Features absent from this table (tld,
# vowel_ratio, consonant_ratio, num_unique_chars, has_www, tld_length,
# whois_country, num_ns_records, has_ipv6, network_features_available) are
# intentionally unmapped: they are weak, contextual, or meta-flags.
_HANDLERS: Dict[str, Tuple[str, Callable[[Any, Dict[str, Any]], Optional[str]]]] = {
    # domain age (deduped)
    "domain_age_days": ("domain_age", _h_domain_age_days),
    "is_newly_registered": ("domain_age", _h_is_newly_registered),
    # whois lifecycle
    "days_until_expiry": ("expiry", _h_days_until_expiry),
    "registration_length_days": ("registration_length", _h_registration_length),
    "registrar_is_common": ("registrar", _h_registrar_is_common),
    "whois_privacy_enabled": ("whois_privacy", _h_whois_privacy),
    # dns
    "has_a_record": ("dns_missing_a", _h_has_a_record),
    "num_a_records": ("many_ips", _h_num_a_records),
    "has_mx_record": ("no_mx", _h_has_mx_record),
    "has_ns_record": ("no_ns", _h_has_ns_record),
    "has_txt_record": ("no_txt", _h_has_txt_record),
    "dns_ttl": ("fast_flux", _h_dns_ttl),
    "is_fast_flux": ("fast_flux", _h_is_fast_flux),
    "dns_resolves": ("no_resolve", _h_dns_resolves),
    # structural
    "tld_is_suspicious": ("suspicious_tld", _h_tld_is_suspicious),
    "contains_phishing_keyword": ("phishing_keyword", _h_contains_phishing_keyword),
    "num_phishing_keywords": ("phishing_keyword", _h_num_phishing_keywords),
    "contains_brand_name": ("brand", _h_contains_brand_name),
    "brand_in_subdomain": ("brand_subdomain", _h_brand_in_subdomain),
    "is_typosquat_candidate": ("typosquat", _h_is_typosquat),
    "typosquat_distance": ("typosquat", _h_typosquat_distance),
    "num_hyphens": ("hyphens", _h_num_hyphens),
    "entropy": ("randomness", _h_entropy),
    "longest_consonant_run": ("randomness", _h_longest_consonant_run),
    "char_repeat_ratio": ("randomness", _h_char_repeat_ratio),
    "hex_ratio": ("randomness", _h_hex_ratio),
    "subdomain_entropy": ("subdomain_randomness", _h_subdomain_entropy),
    "num_digits": ("digits", _h_num_digits),
    "digit_ratio": ("digits", _h_digit_ratio),
    "ratio_digits_to_letters": ("digits", _h_ratio_digits_to_letters),
    "has_ip_in_name": ("ip_disguise", _h_has_ip_in_name),
    "is_punycode": ("punycode", _h_is_punycode),
    "sld_is_numeric": ("numeric_sld", _h_sld_is_numeric),
    "num_subdomains": ("subdomain_depth", _h_num_subdomains),
    "num_dots": ("subdomain_depth", _h_num_dots),
    "domain_length": ("length", _h_domain_length),
    "sld_length": ("length", _h_sld_length),
    "num_special_chars": ("special_chars", _h_num_special_chars),
}


def has_reason_for(feature_name: str) -> bool:
    """Whether a feature can ever produce a reason (used by explainer/tests)."""
    return feature_name in _HANDLERS


def map_reason(
    feature_name: str,
    feature_value: Any,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[Reason]:
    """
    Render a single feature/value into a Reason, or None if no reason applies.
    """
    entry = _HANDLERS.get(feature_name)
    if entry is None:
        return None
    category, handler = entry
    ctx = context or {}
    try:
        text = handler(feature_value, ctx)
    except Exception:
        return None
    if not text:
        return None
    return Reason(feature=feature_name, category=category, text=text)


def build_reasons(
    candidates: Iterable[Tuple[str, Any]],
    context: Optional[Dict[str, Any]] = None,
    top_k: int = 3,
) -> List[str]:
    """
    Render an ordered list of candidate (feature_name, feature_value) pairs into
    up to `top_k` reason strings.

    `candidates` MUST already be ordered by importance (most incriminating
    first) — the explainer sorts by SHAP contribution before calling this.
    Reasons are de-duplicated by category so the same underlying signal is only
    shown once, and rendering stops once `top_k` reasons are collected.
    """
    ctx = context or {}
    reasons: List[str] = []
    seen_categories: set[str] = set()

    for feature_name, feature_value in candidates:
        if len(reasons) >= top_k:
            break
        reason = map_reason(feature_name, feature_value, ctx)
        if reason is None or reason.category in seen_categories:
            continue
        seen_categories.add(reason.category)
        reasons.append(reason.text)

    return reasons