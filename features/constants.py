"""
features/constants.py
======================

Canonical feature metadata for ADIS.

This module is the single source of truth for:

    * the exact order of the 48-element feature vector,
    * which features are structural vs. network,
    * which features are categorical (string-encoded) vs. boolean,
    * the default value to use for every network feature when a live
      DNS / WHOIS lookup fails or is skipped.

CRITICAL: ``FEATURE_NAMES`` defines the exact column order used at both
training time (``ml/training/feature_builder.py``) and inference time
(``features/assembler.py``). If this order ever changes, previously trained
models become invalid and MUST be retrained. Do not reorder these lists;
only append new features to the end and retrain.

The names here correspond 1:1 to the specification in section 6 of the
project blueprint (features 1-30 structural, 31-48 network).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

# ---------------------------------------------------------------------------
# Structural / lexical features (blueprint 6.1, features 1-30).
# Computed from the domain string alone. Always available, zero latency.
# ---------------------------------------------------------------------------
STRUCTURAL_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "domain_length",            # 1
    "sld_length",               # 2
    "tld",                      # 3  (categorical, encoded to int)
    "tld_is_suspicious",        # 4  (bool)
    "num_dots",                 # 5
    "num_hyphens",              # 6
    "num_digits",               # 7
    "digit_ratio",              # 8
    "entropy",                  # 9
    "vowel_ratio",              # 10
    "consonant_ratio",          # 11
    "longest_consonant_run",    # 12
    "num_unique_chars",         # 13
    "char_repeat_ratio",        # 14
    "has_ip_in_name",           # 15 (bool)
    "is_punycode",              # 16 (bool)
    "num_subdomains",           # 17
    "has_www",                  # 18 (bool)
    "sld_is_numeric",           # 19 (bool)
    "contains_phishing_keyword",# 20 (bool)
    "num_phishing_keywords",    # 21
    "contains_brand_name",      # 22 (bool)
    "brand_in_subdomain",       # 23 (bool)
    "typosquat_distance",       # 24
    "is_typosquat_candidate",   # 25 (bool)
    "hex_ratio",                # 26
    "num_special_chars",        # 27
    "subdomain_entropy",        # 28
    "tld_length",               # 29
    "ratio_digits_to_letters",  # 30
)

# ---------------------------------------------------------------------------
# Network features (blueprint 6.2, features 31-48).
# Require live DNS / WHOIS lookups. May fail or time out -> default value.
# ---------------------------------------------------------------------------
NETWORK_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "domain_age_days",              # 31
    "is_newly_registered",          # 32 (bool)
    "days_until_expiry",            # 33
    "registration_length_days",     # 34
    "registrar_is_common",          # 35 (bool)
    "whois_country",                # 36 (categorical, encoded to int)
    "whois_privacy_enabled",        # 37 (bool)
    "has_a_record",                 # 38 (bool)
    "num_a_records",                # 39
    "has_mx_record",                # 40 (bool)
    "has_ns_record",                # 41 (bool)
    "num_ns_records",               # 42
    "has_txt_record",               # 43 (bool)
    "dns_ttl",                      # 44
    "is_fast_flux",                 # 45 (bool)
    "has_ipv6",                     # 46 (bool)
    "dns_resolves",                 # 47 (bool)
    "network_features_available",   # 48 (bool, meta-flag)
)

# Full ordered vector: structural first, then network. THIS ORDER IS FIXED.
FEATURE_NAMES: Final[tuple[str, ...]] = (
    STRUCTURAL_FEATURE_NAMES + NETWORK_FEATURE_NAMES
)

FEATURE_COUNT: Final[int] = len(FEATURE_NAMES)  # 48

# ---------------------------------------------------------------------------
# Categorical features. These arrive as strings and must be integer-encoded
# before entering the numeric feature vector (see assembler._encode_categorical).
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("tld", "whois_country")

# ---------------------------------------------------------------------------
# Boolean features. The assembler coerces these to 0.0 / 1.0. This set is
# informational / for validation; coercion in the assembler is type-driven
# (any Python bool becomes 0/1) so a missing entry here is not fatal.
# ---------------------------------------------------------------------------
BOOLEAN_FEATURES: Final[frozenset[str]] = frozenset(
    {
        "tld_is_suspicious",
        "has_ip_in_name",
        "is_punycode",
        "has_www",
        "sld_is_numeric",
        "contains_phishing_keyword",
        "contains_brand_name",
        "brand_in_subdomain",
        "is_typosquat_candidate",
        "is_newly_registered",
        "registrar_is_common",
        "whois_privacy_enabled",
        "has_a_record",
        "has_mx_record",
        "has_ns_record",
        "has_txt_record",
        "is_fast_flux",
        "has_ipv6",
        "dns_resolves",
        "network_features_available",
    }
)

# ---------------------------------------------------------------------------
# Default values for every network feature, applied when a DNS/WHOIS lookup
# fails, times out, or network extraction is skipped entirely.
# Values taken directly from the "Default if Unavailable" column of the
# blueprint feature table (section 6.2).
# ---------------------------------------------------------------------------
_NETWORK_FEATURE_DEFAULTS: dict[str, object] = {
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

# Read-only view so callers cannot accidentally mutate the shared defaults.
NETWORK_FEATURE_DEFAULTS: Final[MappingProxyType] = MappingProxyType(
    _NETWORK_FEATURE_DEFAULTS
)

# Fallback value for a categorical when nothing is provided at all.
CATEGORICAL_UNKNOWN: Final[str] = "unknown"


# ---------------------------------------------------------------------------
# Integrity checks. These run at import time so a mistake in the lists above
# fails loudly and immediately rather than silently corrupting the model.
# ---------------------------------------------------------------------------
def _validate() -> None:
    assert len(STRUCTURAL_FEATURE_NAMES) == 30, (
        f"expected 30 structural features, got {len(STRUCTURAL_FEATURE_NAMES)}"
    )
    assert len(NETWORK_FEATURE_NAMES) == 18, (
        f"expected 18 network features, got {len(NETWORK_FEATURE_NAMES)}"
    )
    assert FEATURE_COUNT == 48, f"expected 48 total features, got {FEATURE_COUNT}"
    assert len(set(FEATURE_NAMES)) == FEATURE_COUNT, "duplicate feature name detected"

    # Every network feature must have a default.
    missing_defaults = set(NETWORK_FEATURE_NAMES) - set(NETWORK_FEATURE_DEFAULTS)
    assert not missing_defaults, f"network features missing defaults: {missing_defaults}"

    # Defaults must not reference unknown features.
    extra_defaults = set(NETWORK_FEATURE_DEFAULTS) - set(NETWORK_FEATURE_NAMES)
    assert not extra_defaults, f"defaults for non-network features: {extra_defaults}"

    # Categoricals must actually exist in the vector.
    unknown_categoricals = set(CATEGORICAL_FEATURES) - set(FEATURE_NAMES)
    assert not unknown_categoricals, (
        f"unknown categorical features: {unknown_categoricals}"
    )


_validate()


__all__ = [
    "STRUCTURAL_FEATURE_NAMES",
    "NETWORK_FEATURE_NAMES",
    "FEATURE_NAMES",
    "FEATURE_COUNT",
    "CATEGORICAL_FEATURES",
    "BOOLEAN_FEATURES",
    "NETWORK_FEATURE_DEFAULTS",
    "CATEGORICAL_UNKNOWN",
]
