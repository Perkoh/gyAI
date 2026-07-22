"""
gyAI — Structural / Lexical Feature Extraction
==============================================

Implements all 30 structural features from Section 6.1 of the project
blueprint. Every feature is computed purely from the domain string:
no network calls, no I/O, microsecond execution.

Public API
----------
    extract_structural_features(domain: str) -> dict
        Returns an ordered dict of all 30 features, keyed by the exact
        feature names (and order) defined in the blueprint.

    STRUCTURAL_FEATURE_NAMES : tuple[str, ...]
        Canonical feature order — must match training order in
        ``features/constants.py``.

Notes
-----
* The ``tld`` feature is returned as a raw string here (e.g. ``"com"``,
  ``"co.uk"``). Categorical encoding happens in the training
  preprocessing step (``ml/training/preprocess.py``), never here, so the
  same code path runs at training and inference time.
* Keyword / brand / TLD lists are imported from ``config.constants``
  when available (per the blueprint build order); the blueprint defaults
  are embedded as fallbacks so this module is testable standalone.
* Typosquat distance uses ``python-Levenshtein`` when installed and
  falls back to a pure-Python implementation otherwise.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Tuple, Union

# ---------------------------------------------------------------------------
# Optional dependencies (graceful fallbacks keep the module importable
# in minimal environments, e.g. CI without native wheels).
# ---------------------------------------------------------------------------

try:  # Preferred: accurate PSL-based parsing (blueprint tech stack).
    import tldextract as _tldextract

    # suffix_list_urls=() forces the bundled Public Suffix List snapshot,
    # so no network call ever happens at import or request time.
    _EXTRACTOR = _tldextract.TLDExtract(suffix_list_urls=())
except ImportError:  # pragma: no cover - exercised only without tldextract
    _EXTRACTOR = None

try:  # Preferred: C-accelerated edit distance (blueprint tech stack).
    from Levenshtein import distance as _lev_distance  # type: ignore
except ImportError:  # pragma: no cover
    _lev_distance = None


# ---------------------------------------------------------------------------
# Constant lists — imported from config.constants when present,
# otherwise the exact defaults from blueprint Section 6.1.
# ---------------------------------------------------------------------------

_DEFAULT_SUSPICIOUS_TLDS = frozenset(
    {
        "xyz", "top", "click", "tk", "ml", "ga", "cf", "gq", "pw", "cc",
        "biz", "info", "su", "work", "party", "racing", "date", "faith",
        "science", "men", "accountant", "loan", "stream", "download",
        "gdn", "icu", "monster", "cyou", "bond", "hair", "beauty", "makeup",
    }
)

_DEFAULT_PHISHING_KEYWORDS = (
    "login", "secure", "verify", "account", "update", "bank", "signin",
    "confirm", "password", "reset", "billing", "support", "service",
    "invoice", "payment", "checkout", "shipping", "tracking",
    "notification", "alert", "suspended", "locked", "unusual", "activity",
    "webscr", "ebayisapi", "appleid", "icloud", "recovery",
    "authenticate", "credential",
)

_DEFAULT_BRAND_NAMES = (
    "paypal", "google", "amazon", "apple", "microsoft", "facebook",
    "netflix", "twitter", "instagram", "youtube", "linkedin", "ebay",
    "dropbox", "adobe", "chase", "wellsfargo", "bankofamerica",
    "citibank", "dhl", "fedex", "usps", "steam", "discord", "slack",
    "whatsapp", "telegram", "coinbase", "binance", "metamask", "robinhood",
)

# SLDs of highly popular domains used for typosquat detection.
# Replace / extend with the Tranco top-500 SLDs for production
# (config.constants.POPULAR_DOMAINS). Order does not matter.
_DEFAULT_POPULAR_DOMAINS = (
    "google", "youtube", "facebook", "instagram", "twitter", "whatsapp",
    "wikipedia", "yahoo", "amazon", "reddit", "tiktok", "linkedin",
    "netflix", "microsoft", "bing", "office", "live", "pinterest",
    "twitch", "ebay", "apple", "icloud", "adobe", "spotify", "paypal",
    "github", "gitlab", "stackoverflow", "wordpress", "blogspot",
    "tumblr", "quora", "medium", "dropbox", "salesforce", "zoom",
    "slack", "discord", "telegram", "signal", "snapchat", "vimeo",
    "soundcloud", "shopify", "etsy", "aliexpress", "alibaba", "walmart",
    "target", "bestbuy", "homedepot", "costco", "ikea", "nike",
    "booking", "airbnb", "expedia", "tripadvisor", "uber", "lyft",
    "doordash", "instacart", "chase", "wellsfargo", "bankofamerica",
    "citibank", "capitalone", "americanexpress", "visa", "mastercard",
    "coinbase", "binance", "kraken", "robinhood", "fidelity", "vanguard",
    "schwab", "hsbc", "barclays", "santander", "revolut", "stripe",
    "venmo", "zelle", "cashapp", "westernunion", "dhl", "fedex", "ups",
    "usps", "royalmail", "canva", "figma", "notion", "trello", "asana",
    "atlassian", "jira", "confluence", "zendesk", "hubspot", "mailchimp",
    "godaddy", "namecheap", "cloudflare", "digitalocean", "heroku",
    "vercel", "netlify", "wix", "squarespace", "weebly", "mozilla",
    "firefox", "opera", "brave", "duckduckgo", "baidu", "yandex",
    "naver", "daum", "rakuten", "mercadolibre", "flipkart", "myntra",
    "snapdeal", "jumia", "konga", "takealot", "vodafone", "orange",
    "tmobile", "verizon", "comcast", "xfinity", "spectrum", "bbc",
    "cnn", "nytimes", "theguardian", "reuters", "bloomberg", "forbes",
    "wsj", "washingtonpost", "aljazeera", "espn", "nfl", "nba", "fifa",
    "steamcommunity", "steampowered", "epicgames", "roblox", "minecraft",
    "playstation", "xbox", "nintendo", "ea", "ubisoft", "riotgames",
    "leagueoflegends", "fortnite", "gmail", "outlook", "hotmail",
    "protonmail", "zoho", "yahoo", "aol", "gmx", "fastmail", "duolingo",
    "coursera", "udemy", "edx", "khanacademy", "chegg", "grammarly",
    "translate", "maps", "drive", "docs", "onedrive", "sharepoint",
    "teams", "skype", "webex", "gotomeeting", "eventbrite", "meetup",
    "patreon", "kickstarter", "gofundme", "indiegogo", "hulu", "disney",
    "disneyplus", "hbomax", "primevideo", "peacocktv", "crunchyroll",
    "imdb", "rottentomatoes", "goodreads", "audible", "kindle",
    "scribd", "archive", "pornhub", "onlyfans", "craigslist", "zillow",
    "realtor", "indeed", "glassdoor", "monster", "ziprecruiter",
    "upwork", "fiverr", "freelancer", "behance", "dribbble",
    "deviantart", "unsplash", "pexels", "shutterstock", "gettyimages",
    "weather", "accuweather", "webmd", "mayoclinic", "nih", "who",
    "irs", "gov", "usa", "europa", "un", "whitehouse", "nasa",
    "spacex", "tesla", "toyota", "honda", "ford", "bmw", "mercedes",
    "samsung", "sony", "lg", "huawei", "xiaomi", "oneplus", "oppo",
    "lenovo", "dell", "hp", "asus", "acer", "intel", "amd", "nvidia",
    "oracle", "ibm", "sap", "cisco", "vmware", "redhat", "ubuntu",
    "debian", "python", "nodejs", "npmjs", "pypi", "docker",
    "kubernetes", "openai", "anthropic", "claude", "chatgpt",
    "huggingface", "kaggle", "colab", "jupyter",
)

try:  # Blueprint build order creates config/constants.py before this file.
    from config.constants import (  # type: ignore
        TOP_BRANDS,
        PHISHING_KEYWORDS,
        TOP_500_DOMAINS,
        SUSPICIOUS_TLDS,
    )
except ImportError:
    SUSPICIOUS_TLDS = _DEFAULT_SUSPICIOUS_TLDS
    PHISHING_KEYWORDS = _DEFAULT_PHISHING_KEYWORDS
    TOP_BRANDS = _DEFAULT_BRAND_NAMES
    TOP_500_DOMAINS = _DEFAULT_POPULAR_DOMAINS

# Normalise once at import time (strip leading dots, lowercase, dedupe).
_SUSPICIOUS_TLDS: frozenset = frozenset(t.lower().lstrip(".") for t in SUSPICIOUS_TLDS)
_PHISHING_KEYWORDS: Tuple[str, ...] = tuple(dict.fromkeys(k.lower() for k in PHISHING_KEYWORDS))
_BRAND_NAMES: Tuple[str, ...] = tuple(dict.fromkeys(b.lower() for b in TOP_BRANDS))
_POPULAR_DOMAINS: Tuple[str, ...] = tuple(dict.fromkeys(d.lower() for d in TOP_500_DOMAINS))


# ---------------------------------------------------------------------------
# Canonical feature order — MUST match blueprint Section 6.1 (#1–#30)
# and the training order in features/constants.py.
# ---------------------------------------------------------------------------

STRUCTURAL_FEATURE_NAMES: Tuple[str, ...] = (
    "domain_length",              # 1
    "sld_length",                 # 2
    "tld",                        # 3  (raw string; encoded in preprocessing)
    "tld_is_suspicious",          # 4
    "num_dots",                   # 5
    "num_hyphens",                # 6
    "num_digits",                 # 7
    "digit_ratio",                # 8
    "entropy",                    # 9
    "vowel_ratio",                # 10
    "consonant_ratio",            # 11
    "longest_consonant_run",      # 12
    "num_unique_chars",           # 13
    "char_repeat_ratio",          # 14
    "has_ip_in_name",             # 15
    "is_punycode",                # 16
    "num_subdomains",             # 17
    "has_www",                    # 18
    "sld_is_numeric",             # 19
    "contains_phishing_keyword",  # 20
    "num_phishing_keywords",      # 21
    "contains_brand_name",        # 22
    "brand_in_subdomain",         # 23
    "typosquat_distance",         # 24
    "is_typosquat_candidate",     # 25
    "hex_ratio",                  # 26
    "num_special_chars",          # 27
    "subdomain_entropy",          # 28
    "tld_length",                 # 29
    "ratio_digits_to_letters",    # 30
)

NUM_STRUCTURAL_FEATURES: int = len(STRUCTURAL_FEATURE_NAMES)  # == 30

_VOWELS = frozenset("aeiou")
_CONSONANTS = frozenset("bcdfghjklmnpqrstvwxyz")
_HEX_CHARS = frozenset("0123456789abcdef")

# IPv4-shaped labels, dot- OR hyphen-separated (e.g. 192-168-0-1.com).
_IP_PATTERN = re.compile(
    r"(?:^|[.\-])"
    r"(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
    r"(?:[.\-](25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}"
    r"(?:$|[.\-])"
)

_SCHEME_PATTERN = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_domain(raw: str) -> str:
    """Normalise arbitrary input to a bare, lowercase domain string.

    Defensively strips scheme, credentials, port, path, query, fragment
    and trailing dots so callers can pass either ``example.com`` or a
    full URL. The extension should already send only the hostname
    (privacy principle), but the API must not break if it doesn't.
    """
    domain = (raw or "").strip().lower()
    domain = _SCHEME_PATTERN.sub("", domain)
    domain = domain.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    if "@" in domain:  # strip userinfo, e.g. user@evil.com
        domain = domain.rsplit("@", 1)[-1]
    # Strip port, but not the closing bracket of an IPv6 literal.
    if not domain.startswith("["):
        domain = domain.split(":", 1)[0]
    return domain.strip(".")


# Minimal multi-part suffix set for the no-tldextract fallback parser only.
_FALLBACK_MULTI_SUFFIXES = frozenset(
    {
        "co.uk", "org.uk", "ac.uk", "gov.uk", "me.uk", "net.uk",
        "com.au", "net.au", "org.au", "co.nz", "co.za", "com.br",
        "com.mx", "com.ar", "co.in", "net.in", "org.in", "co.jp",
        "ne.jp", "or.jp", "com.cn", "net.cn", "org.cn", "com.hk",
        "com.sg", "com.tr", "com.ng", "com.gh", "com.eg", "co.ke",
        "com.sa", "com.ua", "com.pl", "com.ru",
    }
)


def split_domain(domain: str) -> Tuple[str, str, str]:
    """Split a normalised domain into ``(subdomain, sld, tld)``.

    Uses tldextract's bundled Public Suffix List when available (the
    production path); otherwise falls back to a small built-in
    multi-part suffix table. All parts are returned without dots on
    either end; ``tld`` may itself contain a dot (e.g. ``co.uk``).
    """
    if _EXTRACTOR is not None:
        parts = _EXTRACTOR(domain)
        return parts.subdomain, parts.domain, parts.suffix

    labels = domain.split(".")
    if len(labels) < 2:
        return "", domain, ""
    if len(labels) >= 3 and ".".join(labels[-2:]) in _FALLBACK_MULTI_SUFFIXES:
        tld = ".".join(labels[-2:])
        sld = labels[-3]
        sub = ".".join(labels[:-3])
    else:
        tld = labels[-1]
        sld = labels[-2]
        sub = ".".join(labels[:-2])
    return sub, sld, tld


def shannon_entropy(text: str) -> float:
    """Shannon entropy (bits/char) of a string; 0.0 for empty input."""
    if not text:
        return 0.0
    length = len(text)
    counts = Counter(text)
    entropy = -sum(
        (count / length) * math.log2(count / length) for count in counts.values()
    )
    return entropy if entropy > 0.0 else 0.0  # normalise -0.0 -> 0.0


def longest_consonant_run(text: str) -> int:
    """Length of the longest consecutive run of ASCII consonants."""
    longest = current = 0
    for char in text:
        if char in _CONSONANTS:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest


def _levenshtein_py(a: str, b: str, cutoff: int) -> int:
    """Pure-Python banded Levenshtein with early exit above ``cutoff``.

    Fallback for environments without python-Levenshtein. Returns
    ``cutoff + 1`` as soon as the true distance provably exceeds it.
    """
    if a == b:
        return 0
    len_a, len_b = len(a), len(b)
    if abs(len_a - len_b) > cutoff:
        return cutoff + 1
    if len_a > len_b:  # ensure b is the longer string
        a, b, len_a, len_b = b, a, len_b, len_a
    previous = list(range(len_a + 1))
    for i, char_b in enumerate(b, start=1):
        current = [i]
        row_min = i
        for j, char_a in enumerate(a, start=1):
            cost = 0 if char_a == char_b else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            if value < row_min:
                row_min = value
        if row_min > cutoff:
            return cutoff + 1
        previous = current
    return previous[-1]


def min_edit_distance(sld: str, candidates: Tuple[str, ...] = _POPULAR_DOMAINS) -> int:
    """Minimum Levenshtein distance from ``sld`` to any popular SLD.

    Distance 0 means the SLD *is* a popular domain (e.g. ``google``).
    Returns a large sentinel (99) for an empty SLD.
    """
    if not sld:
        return 99
    best = 99
    for candidate in candidates:
        if _lev_distance is not None:
            dist = _lev_distance(sld, candidate)
        else:
            dist = _levenshtein_py(sld, candidate, cutoff=min(best, 10))
        if dist < best:
            best = dist
            if best == 0:
                break
    return best


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_structural_features(
    domain: str,
) -> Dict[str, Union[int, float, str]]:
    """Extract all 30 structural/lexical features from a raw domain string.

    Parameters
    ----------
    domain:
        Raw domain (``paypal-secure-login.xyz``) or full URL — input is
        normalised first. Never triggers any network I/O.

    Returns
    -------
    dict
        Keys exactly match :data:`STRUCTURAL_FEATURE_NAMES`, in order
        (Python dicts preserve insertion order). Booleans are returned
        as ``int`` (0/1) so the vector is directly model-consumable;
        ``tld`` is a raw string encoded later in preprocessing.
    """
    full = normalize_domain(domain)
    subdomain, sld, tld = split_domain(full)

    sld_len = len(sld)
    safe_sld_len = max(sld_len, 1)  # guard division for degenerate input

    num_digits = sum(char.isdigit() for char in full)
    num_letters = sum(char.isalpha() for char in full)

    vowel_count = sum(char in _VOWELS for char in sld)
    consonant_count = sum(char in _CONSONANTS for char in sld)

    unique_chars = len(set(sld))

    # Keywords are matched against the SLD and subdomain portions only
    # (never the TLD), per feature #20.
    keyword_haystack = f"{subdomain}.{sld}" if subdomain else sld
    keywords_found = [kw for kw in _PHISHING_KEYWORDS if kw in keyword_haystack]

    brand_in_sld = any(brand in sld for brand in _BRAND_NAMES)
    brand_in_sub = bool(subdomain) and any(
        brand in subdomain for brand in _BRAND_NAMES
    )

    typo_distance = min_edit_distance(sld)

    # Alphanumeric SLDs only for hex_ratio; empty SLD -> 0.0.
    hex_count = sum(char in _HEX_CHARS for char in sld)

    # Punycode-encoded labels start with "xn--" (any label, feature #16).
    is_punycode = any(label.startswith("xn--") for label in full.split("."))

    features: Dict[str, Union[int, float, str]] = {
        # 1 — total character count of the full domain
        "domain_length": len(full),
        # 2 — length of the second-level domain
        "sld_length": sld_len,
        # 3 — raw TLD string (categorical; encoded in preprocessing)
        "tld": tld,
        # 4 — TLD is in the high-abuse list
        "tld_is_suspicious": int(tld in _SUSPICIOUS_TLDS),
        # 5 — dots in the full domain
        "num_dots": full.count("."),
        # 6 — hyphens in the full domain
        "num_hyphens": full.count("-"),
        # 7 — digit characters in the full domain
        "num_digits": num_digits,
        # 8 — num_digits / sld_length (per spec)
        "digit_ratio": num_digits / safe_sld_len,
        # 9 — Shannon entropy of the SLD
        "entropy": shannon_entropy(sld),
        # 10 — vowel_count / sld_length
        "vowel_ratio": vowel_count / safe_sld_len,
        # 11 — consonant_count / sld_length
        "consonant_ratio": consonant_count / safe_sld_len,
        # 12 — longest consecutive consonant sequence in SLD
        "longest_consonant_run": longest_consonant_run(sld),
        # 13 — distinct characters in the SLD
        "num_unique_chars": unique_chars,
        # 14 — (sld_length - num_unique_chars) / sld_length
        "char_repeat_ratio": (sld_len - unique_chars) / safe_sld_len,
        # 15 — IPv4-shaped name, dot- or hyphen-separated
        "has_ip_in_name": int(bool(_IP_PATTERN.search(full))),
        # 16 — internationalised (punycode) domain name
        "is_punycode": int(is_punycode),
        # 17 — count of subdomain labels
        "num_subdomains": len(subdomain.split(".")) if subdomain else 0,
        # 18 — starts with "www."
        "has_www": int(full.startswith("www.")),
        # 19 — SLD is entirely digits
        "sld_is_numeric": int(bool(sld) and sld.isdigit()),
        # 20 — any phishing keyword in SLD or subdomains
        "contains_phishing_keyword": int(bool(keywords_found)),
        # 21 — count of distinct phishing keywords present
        "num_phishing_keywords": len(keywords_found),
        # 22 — known brand name inside the SLD
        "contains_brand_name": int(brand_in_sld),
        # 23 — brand in subdomain but NOT in SLD (classic phishing pattern)
        "brand_in_subdomain": int(brand_in_sub and not brand_in_sld),
        # 24 — min Levenshtein distance to popular domains
        "typosquat_distance": typo_distance,
        # 25 — distance 1–2 = likely typosquat. Distance 0 is an exact
        #      match to a popular domain (the real site), not a typo.
        "is_typosquat_candidate": int(1 <= typo_distance <= 2),
        # 26 — ratio of hex characters (0-9, a-f) in the SLD
        "hex_ratio": hex_count / safe_sld_len,
        # 27 — non-alphanumeric, non-hyphen, non-dot characters
        "num_special_chars": sum(
            not (char.isalnum() or char in ".-") for char in full
        ),
        # 28 — Shannon entropy of the subdomain portion (dots excluded)
        "subdomain_entropy": shannon_entropy(subdomain.replace(".", "")),
        # 29 — character length of the TLD
        "tld_length": len(tld),
        # 30 — num_digits / (num_letters + 1) on the full domain
        "ratio_digits_to_letters": num_digits / (num_letters + 1),
    }

    return features


def structural_feature_values(domain: str) -> List[Union[int, float, str]]:
    """Feature values as a list in canonical order (assembler helper)."""
    features = extract_structural_features(domain)
    return [features[name] for name in STRUCTURAL_FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Manual smoke test:  python -m features.structural
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import json
    import time

    samples = [
        "google.com",
        "www.github.com",
        "paypal-secure-login.xyz",
        "paypal.some-attacker.com",
        "secure-login-paypa1.xyz",
        "xn--pypal-4ve.com",
        "192-168-0-1.com",
        "a8f3k2j9q1x7.top",
        "bbc.co.uk",
    ]
    for sample in samples:
        start = time.perf_counter()
        result = extract_structural_features(sample)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"\n=== {sample}  ({elapsed_ms:.3f} ms) ===")
        print(json.dumps(result, indent=2))
        assert list(result) == list(STRUCTURAL_FEATURE_NAMES)
        assert len(result) == NUM_STRUCTURAL_FEATURES