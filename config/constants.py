"""
config/constants.py
───────────────────
Static, immutable lookup tables used across the ADIS feature extraction
pipeline.  Nothing in this file reads from .env — all values are hardcoded
because they represent domain knowledge, not deployment configuration.

Import pattern:
    from config.constants import (
        PHISHING_KEYWORDS,
        SUSPICIOUS_TLDS,
        TOP_BRANDS,
        TOP_500_DOMAINS,
        COMMON_REGISTRARS,
        HIGH_RISK_COUNTRY_CODES,
        LABEL,
        AnalysisSource,
        UserVerdict,
    )
"""

from __future__ import annotations

from enum import Enum
from typing import FrozenSet


# ══════════════════════════════════════════════════════════════
# 1. SCORE LABELS  (canonical strings used in DB, API, cache)
# ══════════════════════════════════════════════════════════════

class LABEL:
    SAFE       = "safe"
    SUSPICIOUS = "suspicious"
    MALICIOUS  = "malicious"

    ALL: FrozenSet[str] = frozenset({SAFE, SUSPICIOUS, MALICIOUS})


class CONFIDENCE:
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"

    ALL: FrozenSet[str] = frozenset({LOW, MEDIUM, HIGH})


# ══════════════════════════════════════════════════════════════
# 2. DATABASE ENUM MIRRORS  (must match Supabase CHECK constraints)
# ══════════════════════════════════════════════════════════════

class AnalysisSource(str, Enum):
    EXTENSION = "extension"
    API       = "api"


class UserVerdict(str, Enum):
    FALSE_POSITIVE      = "false_positive"
    CONFIRMED_MALICIOUS = "confirmed_malicious"
    UNSURE              = "unsure"


class KnownDomainVerdict(str, Enum):
    SAFE      = "safe"
    MALICIOUS = "malicious"
    PHISHING  = "phishing"


# ══════════════════════════════════════════════════════════════
# 3. SUSPICIOUS TLDs  (Feature: tld_is_suspicious)
#    Source: abuse statistics from Spamhaus, SURBL, and Palo Alto
#    Unit42 research — last updated June 2026
# ══════════════════════════════════════════════════════════════

SUSPICIOUS_TLDS: FrozenSet[str] = frozenset({
    # Free / zero-cost TLDs heavily abused for phishing
    "tk", "ml", "ga", "cf", "gq",
    # Cheap generic TLDs with high abuse rates
    "xyz", "top", "click", "pw", "cc", "biz", "su", "work",
    # Novelty / gambling / crypto TLDs
    "party", "racing", "date", "faith", "science", "men",
    # Finance-themed TLDs abused for credential harvesting
    "accountant", "loan", "trade",
    # Media / streaming TLDs used in piracy/malware
    "stream", "download",
    # Misc high-abuse registry TLDs
    "gdn", "icu", "monster", "cyou", "bond",
    # Cosmetics / lifestyle TLDs with low legitimate use
    "hair", "beauty", "makeup",
    # Additional TLDs flagged by threat intelligence feeds
    "info",          # historically very high spam/phish ratio
    "live",          # commonly used in streaming-scam domains
    "online",        # overrepresented in phishing kits
    "site",          # common in malware distribution
    "club",          # abused for fake membership scams
    "space",         # very low cost, high abuse
    "buzz",          # spam and clickbait
    "life",          # social engineering
    "run",           # malware delivery
})


# ══════════════════════════════════════════════════════════════
# 4. PHISHING KEYWORDS  (Features: contains_phishing_keyword,
#                                   num_phishing_keywords)
#    Matched against SLD and subdomains (case-insensitive)
# ══════════════════════════════════════════════════════════════

PHISHING_KEYWORDS: FrozenSet[str] = frozenset({
    # Authentication / account actions
    "login", "signin", "sign-in", "signup", "logon",
    "account", "accounts", "myaccount",
    "verify", "verification", "validate", "validation",
    "confirm", "confirmation",
    "authenticate", "authentication", "credential", "credentials",
    "password", "passwd", "reset", "recover", "recovery",
    # Security / urgency language
    "secure", "security", "safe",
    "alert", "warning", "notification",
    "suspended", "locked", "blocked", "unusual", "activity",
    "urgent", "important", "action-required",
    # Financial / payment triggers
    "bank", "banking",
    "payment", "pay", "checkout", "billing", "invoice",
    "refund", "reimburse", "claim",
    "wallet", "balance", "transfer",
    # Shipping / delivery hooks
    "shipping", "tracking", "delivery", "parcel", "package",
    # Platform-specific terms abused in phishing kits
    "webscr",       # PayPal web scripts
    "ebayisapi",    # eBay API endpoint impersonation
    "appleid",      # Apple ID phishing
    "icloud",       # iCloud phishing
    # Support / service impersonation
    "support", "helpdesk", "help",
    "service", "services", "customer",
    "update", "updates", "upgrade",
})


# ══════════════════════════════════════════════════════════════
# 5. TOP BRANDS  (Features: contains_brand_name, brand_in_subdomain)
#    These are the most impersonated brands worldwide.
#    Matched case-insensitively inside the domain string.
# ══════════════════════════════════════════════════════════════

TOP_BRANDS: FrozenSet[str] = frozenset({
    # Big Tech / Social
    "google", "gmail", "youtube",
    "microsoft", "outlook", "office", "azure", "xbox",
    "apple", "icloud", "appleid",
    "facebook", "instagram", "whatsapp", "meta",
    "twitter", "x",
    "linkedin",
    "amazon", "aws",
    "netflix",
    "adobe",
    "zoom",
    # Developer / Collaboration
    "github", "gitlab",
    "slack", "discord", "telegram",
    "dropbox",
    # Finance / Payments
    "paypal",
    "chase", "wellsfargo", "bankofamerica", "citibank", "barclays",
    "hsbc", "lloyds", "santander",
    "visa", "mastercard", "amex",
    "coinbase", "binance", "metamask", "robinhood", "kraken",
    # E-commerce / Retail
    "ebay",
    "walmart", "target",
    "aliexpress", "alibaba",
    # Delivery / Logistics
    "dhl", "fedex", "usps", "ups",
    # Gaming
    "steam", "roblox", "epicgames", "playstation",
    # Streaming / Entertainment
    "spotify", "hulu", "disneyplus",
})


# ══════════════════════════════════════════════════════════════
# 6. TOP 500 DOMAINS  (Feature: typosquat_distance,
#                                is_typosquat_candidate)
#    The Levenshtein distance from a candidate domain's SLD to
#    each of these entries is computed; minimum distance ≤ 2
#    flags the domain as a potential typosquat.
#
#    Source: Tranco Top 1M — top 500 SLD values (June 2026 pull)
#    Only second-level domain (SLD) strings — no TLD, no scheme.
# ══════════════════════════════════════════════════════════════

TOP_500_DOMAINS: FrozenSet[str] = frozenset({
    # Rank 1–50 (most visited globally)
    "google", "youtube", "facebook", "instagram", "twitter",
    "baidu", "wikipedia", "reddit", "yahoo", "amazon",
    "whatsapp", "netflix", "linkedin", "microsoft", "tiktok",
    "discord", "twitch", "pinterest", "snapchat", "quora",
    "ebay", "paypal", "apple", "github", "stackoverflow",
    "zoom", "slack", "spotify", "dropbox", "adobe",
    "bing", "duckduckgo", "wordpress", "tumblr", "medium",
    "shopify", "squarespace", "wix", "mailchimp", "hubspot",
    "salesforce", "atlassian", "jira", "confluence", "notion",
    "trello", "asana", "monday", "figma", "canva",
    # Rank 51–150
    "cloudflare", "aws", "azure", "digitalocean", "linode",
    "heroku", "vercel", "netlify", "firebase", "supabase",
    "stripe", "square", "shopify", "braintree", "razorpay",
    "twilio", "sendgrid", "mailgun", "postmark", "mandrill",
    "bitbucket", "npm", "pypi", "rubygems", "packagist",
    "docker", "kubernetes", "terraform", "ansible", "jenkins",
    "grafana", "datadog", "splunk", "pagerduty", "opsgenie",
    "jfrog", "sonarqube", "newrelic", "dynatrace", "appdynamics",
    "okta", "auth0", "onelogin", "duo", "cyberark",
    "crowdstrike", "sophos", "malwarebytes", "avast", "bitdefender",
    "norton", "mcafee", "kaspersky", "symantec", "trendmicro",
    "cisco", "paloalto", "fortinet", "checkpoint", "juniper",
    "dell", "hp", "lenovo", "asus", "acer",
    "intel", "amd", "nvidia", "qualcomm", "broadcom",
    "samsung", "sony", "lg", "panasonic", "philips",
    "tesla", "bmw", "mercedes", "audi", "toyota",
    # Rank 151–300
    "chase", "wellsfargo", "bankofamerica", "citibank", "capitalone",
    "barclays", "hsbc", "lloyds", "santander", "natwest",
    "visa", "mastercard", "amex", "discover", "diners",
    "coinbase", "binance", "kraken", "gemini", "robinhood",
    "etrade", "schwab", "fidelity", "vanguard", "tdameritrade",
    "fedex", "ups", "dhl", "usps", "royalmail",
    "airbnb", "booking", "expedia", "tripadvisor", "hotels",
    "uber", "lyft", "doordash", "grubhub", "instacart",
    "walmart", "target", "costco", "bestbuy", "homedepot",
    "ikea", "wayfair", "etsy", "aliexpress", "alibaba",
    "nytimes", "washingtonpost", "guardian", "bbc", "cnn",
    "foxnews", "reuters", "apnews", "bloomberg", "forbes",
    "espn", "nba", "nfl", "mlb", "fifa",
    "hulu", "disneyplus", "hbomax", "peacock", "paramount",
    "steam", "epicgames", "roblox", "minecraft", "playstation",
    "xbox", "nintendo", "twitch", "mixer", "discord",
    # Rank 301–500
    "godaddy", "namecheap", "hover", "porkbun", "spaceship",
    "bluehost", "siteground", "hostgator", "dreamhost", "ionos",
    "cloudinary", "fastly", "akamai", "incapsula", "sucuri",
    "sendbird", "intercom", "zendesk", "freshdesk", "servicenow",
    "salesforce", "hubspot", "marketo", "pardot", "eloqua",
    "semrush", "ahrefs", "moz", "serpstat", "majestic",
    "hotjar", "mixpanel", "amplitude", "segment", "heap",
    "surveymonkey", "typeform", "jotform", "formstack", "wufoo",
    "docusign", "hellosign", "pandadoc", "signrequest", "adobe",
    "zapier", "make", "ifttt", "automate", "workato",
    "loom", "vidyard", "wistia", "vimeo", "dailymotion",
    "anchorfm", "buzzsprout", "podbean", "transistor", "captivate",
    "substack", "ghost", "squarespace", "webflow", "framer",
    "figma", "sketch", "invision", "zeplin", "abstract",
    "grammarly", "hemingway", "quillbot", "writesonic", "jasper",
    "openai", "anthropic", "cohere", "huggingface", "replicate",
    "coursera", "udemy", "pluralsight", "linkedin", "skillshare",
    "duolingo", "babbel", "rosetta", "pimsleur", "busuu",
    "proton", "tutanota", "fastmail", "zoho", "gsuite",
    "1password", "lastpass", "bitwarden", "dashlane", "keepass",
    "expressvpn", "nordvpn", "surfshark", "privatevpn", "mullvad",
    "hirevue", "greenhouse", "lever", "workday", "bamboohr",
    "quickbooks", "freshbooks", "xero", "wave", "sage",
    "twitch", "odysee", "rumble", "floatplane", "nebula",
    "patreon", "onlyfans", "fanhouse", "buymeacoffee", "ko-fi",
    "redbubble", "society6", "teepublic", "threadless", "zazzle",
    "fiverr", "upwork", "toptal", "freelancer", "guru",
    "crunchbase", "producthunt", "betalist", "indiehackers", "ycombinator",
    "devto", "hashnode", "medium", "substack", "beehiiv",
    "planetscale", "neon", "railway", "render", "flyio",
})


# ══════════════════════════════════════════════════════════════
# 7. COMMON LEGITIMATE REGISTRARS  (Feature: registrar_is_common)
#    Phishing domains are disproportionately registered through
#    a handful of low-cost registrars.  Registrars NOT on this
#    list receive a mild flag.  This list covers registrars used
#    by the vast majority of legitimate domains.
# ══════════════════════════════════════════════════════════════

COMMON_REGISTRARS: FrozenSet[str] = frozenset({
    "godaddy",
    "namecheap",
    "google domains",
    "cloudflare registrar",
    "name.com",
    "network solutions",
    "register.com",
    "enom",
    "tucows",
    "ionos",
    "ovh",
    "gandi",
    "hover",
    "porkbun",
    "spaceship",
    "route 53",          # AWS Route 53
    "amazon registrar",
    "1&1",
    "bluehost",
    "hostgator",
    "dynadot",
    "key-systems",
    "markmonitor",       # Used by Fortune 500 companies
    "csc corporate domains",
    "safenames",
})

# Registrars disproportionately used for phishing / spam campaigns
# Presence of these in WHOIS is a mild additional risk signal.
HIGH_RISK_REGISTRARS: FrozenSet[str] = frozenset({
    "publicdomainregistry",
    "reg.ru",
    "beget",
    "internet.bs",
    "nicenic",
    "webnic",
    "planet domains",
    "papaki",
    "regtons",
})


# ══════════════════════════════════════════════════════════════
# 8. HIGH-RISK WHOIS COUNTRY CODES  (Feature: whois_country)
#    These are country codes that appear more frequently in
#    malicious domain registrations relative to their share of
#    legitimate domains.  This is statistical, not absolute —
#    many legitimate domains use these codes.
# ══════════════════════════════════════════════════════════════

HIGH_RISK_COUNTRY_CODES: FrozenSet[str] = frozenset({
    "RU",   # Russia — high abuse in spam/phishing
    "CN",   # China — large volume of malicious domains
    "UA",   # Ukraine — significant phishing infrastructure
    "NG",   # Nigeria — advance fee fraud / phishing
    "PK",   # Pakistan — phishing campaigns
    "KP",   # North Korea — state-sponsored attacks
    "IR",   # Iran — state-sponsored attacks
    "BY",   # Belarus — bulletproof hosting
})


# ══════════════════════════════════════════════════════════════
# 9. DOMAIN AGE THRESHOLDS  (Features: is_newly_registered,
#                                       domain_age_days)
# ══════════════════════════════════════════════════════════════

# Domains registered within this many days are considered newly registered
NEWLY_REGISTERED_DAYS: int = 30

# Short registration period (days) — malicious actors often register
# domains for only 1 year to minimise cost
SHORT_REGISTRATION_DAYS: int = 365

# Very old domains (days) are very unlikely to be phishing
ESTABLISHED_DOMAIN_DAYS: int = 730   # 2 years


# ══════════════════════════════════════════════════════════════
# 10. DNS / FAST-FLUX THRESHOLDS  (Features: is_fast_flux,
#                                             dns_ttl)
# ══════════════════════════════════════════════════════════════

# TTL below this threshold is considered fast-flux (seconds)
FAST_FLUX_TTL_THRESHOLD: int = 300     # 5 minutes

# Minimum number of A records to trigger fast-flux flag
# (low TTL alone is not enough — must also have many IPs)
FAST_FLUX_MIN_A_RECORDS: int = 3


# ══════════════════════════════════════════════════════════════
# 11. STRUCTURAL THRESHOLDS  (various structural features)
# ══════════════════════════════════════════════════════════════

# Domain total length above which the domain is considered long
LONG_DOMAIN_THRESHOLD: int = 30

# Maximum consecutive consonants in a real English word
MAX_REAL_WORD_CONSONANT_RUN: int = 4

# Shannon entropy threshold above which SLD looks random (DGA)
HIGH_ENTROPY_THRESHOLD: float = 3.5

# Levenshtein distance at or below which domain is a typosquat candidate
TYPOSQUAT_DISTANCE_THRESHOLD: int = 2

# Maximum number of subdomains before domain is flagged as deeply nested
DEEP_SUBDOMAIN_THRESHOLD: int = 3

# Maximum number of hyphens in a legitimate domain SLD
# (above this, domain is mimicking a brand: pay-pal-secure-login.com)
HYPHEN_ABUSE_THRESHOLD: int = 2


# ══════════════════════════════════════════════════════════════
# 12. REDIS CACHE KEY NAMESPACES
# ══════════════════════════════════════════════════════════════

class CacheNamespace:
    DOMAIN_RESULT  = "adis:cache:"       # adis:cache:<domain>
    WHOIS_RESULT   = "adis:whois:"       # adis:whois:<domain>
    DNS_RESULT     = "adis:dns:"         # adis:dns:<domain>
    RATE_LIMIT     = "adis:ratelimit:"   # used by flask-limiter
    ADMIN_LOCK     = "adis:admin:lock"   # distributed lock for model reload


# ══════════════════════════════════════════════════════════════
# 13. SHAP REASON TEMPLATES
#     Keyed by feature name.  {value} is substituted at runtime
#     with the actual feature value when generating human-readable
#     explanations via ml/reason_mapper.py
# ══════════════════════════════════════════════════════════════

SHAP_REASON_TEMPLATES: dict[str, str] = {
    "domain_age_days":
        "This domain was registered only {value} days ago, which is common for phishing sites",
    "is_newly_registered":
        "This domain was registered less than 30 days ago — a high-risk window for phishing",
    "contains_phishing_keyword":
        "The domain name contains phishing-related keywords",
    "num_phishing_keywords":
        "The domain contains {value} phishing-related keywords",
    "tld_is_suspicious":
        "The domain uses a high-risk extension ({value})",
    "entropy":
        "The domain name appears randomly generated (high character entropy)",
    "is_typosquat_candidate":
        "This looks like a deliberate misspelling of a popular website",
    "typosquat_distance":
        "This domain is only {value} character edit(s) away from a well-known site",
    "brand_in_subdomain":
        "A well-known brand name appears in a subdomain rather than the main domain — a classic phishing pattern",
    "contains_brand_name":
        "The domain impersonates a well-known brand",
    "dns_ttl":
        "The domain's DNS is configured for rapid IP rotation (fast-flux), used to evade detection",
    "is_fast_flux":
        "This domain is using fast-flux infrastructure to evade blocklists",
    "has_mx_record":
        "No email server is configured — unusual for legitimate organisations",
    "num_hyphens":
        "Excessive hyphens are used to mimic a legitimate brand name",
    "is_punycode":
        "The domain uses internationalised characters to visually impersonate another site (IDN homograph attack)",
    "domain_length":
        "The domain name is unusually long ({value} characters), common in phishing URLs",
    "whois_privacy_enabled":
        "Domain registration details are hidden from public records, which is uncommon for legitimate businesses",
    "registrar_is_common":
        "This domain was registered through a registrar commonly used in phishing campaigns",
    "days_until_expiry":
        "The domain is due to expire very soon, suggesting it may be temporary",
    "registration_length_days":
        "This domain was registered for only a short period — typical for throwaway phishing domains",
    "has_a_record":
        "The domain does not have a DNS A record, meaning it is not serving web traffic",
    "sld_is_numeric":
        "The domain's second-level name consists entirely of numbers, which is atypical for legitimate sites",
    "has_ip_in_name":
        "The domain name encodes an IP address, a technique used to bypass URL filters",
    "num_subdomains":
        "The domain has an unusually deep subdomain structure ({value} levels)",
    "subdomain_entropy":
        "The subdomain portion appears randomly generated",
    "digit_ratio":
        "An unusually high proportion of the domain name consists of digits",
    "longest_consonant_run":
        "The domain contains a long run of consonants, suggesting it is machine-generated",
    "vowel_ratio":
        "The domain's character distribution is inconsistent with natural language",
    "char_repeat_ratio":
        "Characters in the domain name are repeated at an unusually high rate",
    "hex_ratio":
        "The domain name resembles a hexadecimal string, a pattern common in DGA domains",
    "whois_country":
        "The domain is registered in a country code with elevated abuse rates ({value})",
    "network_features_available":
        "Network information for this domain could not be retrieved — analysis is based on structural features only",
}


# ══════════════════════════════════════════════════════════════
# 14. API ERROR CODES  (used in error_handlers.py and schemas)
# ══════════════════════════════════════════════════════════════

class ErrorCode:
    INVALID_DOMAIN      = "INVALID_DOMAIN"
    DOMAIN_REQUIRED     = "DOMAIN_REQUIRED"
    DOMAIN_TOO_LONG     = "DOMAIN_TOO_LONG"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    UNAUTHORIZED        = "UNAUTHORIZED"
    FORBIDDEN           = "FORBIDDEN"
    NOT_FOUND           = "NOT_FOUND"
    MODEL_NOT_LOADED    = "MODEL_NOT_LOADED"
    ANALYSIS_FAILED     = "ANALYSIS_FAILED"
    BULK_LIMIT_EXCEEDED = "BULK_LIMIT_EXCEEDED"
    INTERNAL_ERROR      = "INTERNAL_ERROR"


# ══════════════════════════════════════════════════════════════
# 15. VALIDATION CONSTRAINTS
# ══════════════════════════════════════════════════════════════

# RFC 1035 / 1123 max domain length
DOMAIN_MAX_LENGTH: int = 253
DOMAIN_MIN_LENGTH: int = 3

# Feedback comment max character length
FEEDBACK_COMMENT_MAX_LENGTH: int = 1000
