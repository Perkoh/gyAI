"""Unit tests for the 30 structural / lexical features (blueprint section 6.1).

Assumed interface
-----------------
``features/structural.py`` exposes::

    extract_structural_features(domain: str) -> dict[str, int | float | bool | str]

returning a mapping keyed by the exact feature names from the blueprint. Every
value is computed purely from the domain string (no network), so these tests are
fully deterministic and require no mocking.

If ``extract_structural_features`` is not implemented yet, the tests skip with a
clear message instead of erroring.
"""

from __future__ import annotations

import math

import pytest

from conftest import attr_or_skip, import_or_skip

# The 30 structural feature names, in blueprint order.
STRUCTURAL_FEATURE_NAMES = [
    "domain_length",
    "sld_length",
    "tld",
    "tld_is_suspicious",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "digit_ratio",
    "entropy",
    "vowel_ratio",
    "consonant_ratio",
    "longest_consonant_run",
    "num_unique_chars",
    "char_repeat_ratio",
    "has_ip_in_name",
    "is_punycode",
    "num_subdomains",
    "has_www",
    "sld_is_numeric",
    "contains_phishing_keyword",
    "num_phishing_keywords",
    "contains_brand_name",
    "brand_in_subdomain",
    "typosquat_distance",
    "is_typosquat_candidate",
    "hex_ratio",
    "num_special_chars",
    "subdomain_entropy",
    "tld_length",
    "ratio_digits_to_letters",
]


@pytest.fixture
def extract():
    """Return the structural feature extractor, or skip if unimplemented."""
    module = import_or_skip("features.structural")
    return attr_or_skip(module, "extract_structural_features")


# ---------------------------------------------------------------------------
# Shape / completeness of the returned feature mapping.
# ---------------------------------------------------------------------------
def test_returns_all_thirty_feature_names(extract):
    feats = extract("google.com")
    for name in STRUCTURAL_FEATURE_NAMES:
        assert name in feats, f"missing structural feature: {name}"


def test_no_unexpected_structural_keys(extract):
    feats = extract("google.com")
    unexpected = set(feats) - set(STRUCTURAL_FEATURE_NAMES)
    assert not unexpected, f"unexpected structural feature keys: {sorted(unexpected)}"


def test_structural_extraction_never_raises_on_odd_input(extract):
    # Structural features must always succeed (they gate the whole pipeline).
    for weird in ["a.com", "x" * 200 + ".com", "sub.sub.sub.example.co.uk"]:
        feats = extract(weird)
        assert isinstance(feats, dict)


# ---------------------------------------------------------------------------
# Length / counting features.
# ---------------------------------------------------------------------------
def test_domain_length_counts_full_domain(extract):
    assert extract("sub.example.com")["domain_length"] == len("sub.example.com")


def test_sld_length(extract):
    assert extract("example.com")["sld_length"] == len("example")


def test_num_dots(extract):
    assert extract("a.b.example.com")["num_dots"] == 3


def test_num_hyphens(extract):
    assert extract("pay-pal-secure.com")["num_hyphens"] == 2


def test_num_digits(extract):
    assert extract("abc123.com")["num_digits"] == 3


def test_num_subdomains(extract):
    assert extract("a.b.example.com")["num_subdomains"] == 2
    assert extract("example.com")["num_subdomains"] == 0


def test_tld_and_tld_length(extract):
    feats = extract("google.com")
    assert str(feats["tld"]).lstrip(".") == "com"
    assert feats["tld_length"] == 3
    assert extract("secure.info")["tld_length"] == 4


# ---------------------------------------------------------------------------
# Ratio features.
# ---------------------------------------------------------------------------
def test_digit_ratio(extract):
    # sld "abc123" -> 3 digits / 6 chars = 0.5
    assert extract("abc123.com")["digit_ratio"] == pytest.approx(0.5, abs=1e-6)


def test_vowel_and_consonant_ratio_all_vowels(extract):
    feats = extract("aeiou.com")
    assert feats["vowel_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert feats["consonant_ratio"] == pytest.approx(0.0, abs=1e-6)


def test_consonant_ratio_all_consonants(extract):
    feats = extract("bcdfg.com")
    assert feats["consonant_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert feats["vowel_ratio"] == pytest.approx(0.0, abs=1e-6)


def test_char_repeat_ratio(extract):
    # sld "aaaa": unique=1, len=4 -> (4-1)/4 = 0.75
    feats = extract("aaaa.com")
    assert feats["num_unique_chars"] == 1
    assert feats["char_repeat_ratio"] == pytest.approx(0.75, abs=1e-6)


def test_ratio_digits_to_letters_zero_for_all_letters(extract):
    assert extract("google.com")["ratio_digits_to_letters"] == pytest.approx(0.0, abs=1e-6)


def test_hex_ratio_bounds(extract):
    feats = extract("abcdef.com")  # sld is all hex digits
    assert feats["hex_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= extract("google.com")["hex_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# Entropy / randomness features.
# ---------------------------------------------------------------------------
def test_entropy_zero_for_single_repeated_char(extract):
    # Shannon entropy of a constant string is exactly 0.
    assert extract("aaaa.com")["entropy"] == pytest.approx(0.0, abs=1e-6)


def test_entropy_higher_for_random_looking_sld(extract):
    low = extract("google.com")["entropy"]
    high = extract("x7z9q2w1.com")["entropy"]
    assert high > low


def test_subdomain_entropy_zero_when_no_subdomain(extract):
    assert extract("google.com")["subdomain_entropy"] == pytest.approx(0.0, abs=1e-6)


def test_longest_consonant_run(extract):
    # sld "bcdfg" is five consecutive consonants.
    assert extract("bcdfg.com")["longest_consonant_run"] == 5


# ---------------------------------------------------------------------------
# Boolean / flag features.
# ---------------------------------------------------------------------------
def test_has_ip_in_name(extract):
    assert bool(extract("192-168-0-1.com")["has_ip_in_name"]) is True
    assert bool(extract("google.com")["has_ip_in_name"]) is False


def test_is_punycode(extract):
    assert bool(extract("xn--pypal-4ve.com")["is_punycode"]) is True
    assert bool(extract("google.com")["is_punycode"]) is False


def test_has_www(extract):
    assert bool(extract("www.example.com")["has_www"]) is True
    assert bool(extract("example.com")["has_www"]) is False


def test_sld_is_numeric(extract):
    assert bool(extract("12345.com")["sld_is_numeric"]) is True
    assert bool(extract("google.com")["sld_is_numeric"]) is False


def test_tld_is_suspicious(extract):
    assert bool(extract("free-prize.xyz")["tld_is_suspicious"]) is True
    assert bool(extract("google.com")["tld_is_suspicious"]) is False


def test_num_special_chars_zero_for_clean_domain(extract):
    assert extract("google.com")["num_special_chars"] == 0


# ---------------------------------------------------------------------------
# Phishing / brand signals.
# ---------------------------------------------------------------------------
def test_contains_phishing_keyword(extract):
    feats = extract("secure-login.com")
    assert bool(feats["contains_phishing_keyword"]) is True
    assert feats["num_phishing_keywords"] >= 2  # "secure" and "login"


def test_no_phishing_keyword_on_clean_domain(extract):
    feats = extract("google.com")
    assert bool(feats["contains_phishing_keyword"]) is False
    assert feats["num_phishing_keywords"] == 0


def test_contains_brand_name(extract):
    assert bool(extract("paypal-login.com")["contains_brand_name"]) is True
    assert bool(extract("randomsite123.com")["contains_brand_name"]) is False


def test_brand_in_subdomain(extract):
    # Classic phishing shape: real brand buried in the subdomain.
    feats = extract("paypal.attacker-site.com")
    assert bool(feats["brand_in_subdomain"]) is True


# ---------------------------------------------------------------------------
# Typosquatting.
# ---------------------------------------------------------------------------
def test_typosquat_distance_zero_for_known_top_domain(extract):
    feats = extract("google.com")
    assert feats["typosquat_distance"] == 0
    assert bool(feats["is_typosquat_candidate"]) is False  


def test_typosquat_candidate_for_near_miss(extract):
    feats = extract("gooogle.com")  # one extra 'o'
    assert feats["typosquat_distance"] <= 2
    assert bool(feats["is_typosquat_candidate"]) is True


def test_not_typosquat_for_distant_domain(extract):
    feats = extract("zzqwvblkxmpq12345.com")
    assert feats["typosquat_distance"] > 2
    assert bool(feats["is_typosquat_candidate"]) is False


# ---------------------------------------------------------------------------
# Cross-check: a blatant phishing domain should trip several structural flags.
# ---------------------------------------------------------------------------
def test_phishing_domain_trips_multiple_signals(extract):
    feats = extract("secure-login-paypa1.xyz")
    assert bool(feats["tld_is_suspicious"]) is True
    assert bool(feats["contains_phishing_keyword"]) is True
    assert feats["num_hyphens"] >= 2


def test_safe_domain_is_quiet(extract):
    feats = extract("github.com")
    assert bool(feats["tld_is_suspicious"]) is False
    assert bool(feats["contains_phishing_keyword"]) is False
    assert bool(feats["has_ip_in_name"]) is False
    assert bool(feats["is_punycode"]) is False


# ---------------------------------------------------------------------------
# Independent entropy reference (guards against a subtly wrong implementation).
# ---------------------------------------------------------------------------
def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def test_entropy_matches_reference_for_word(extract):
    expected = _shannon_entropy("google")
    assert extract("google.com")["entropy"] == pytest.approx(expected, abs=1e-3)
