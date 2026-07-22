"""Unit tests for the 18 network features (blueprint section 6.2).

Assumed interface
-----------------
``features/network.py`` exposes::

    extract_network_features(domain: str, timeout: float = 3.0) -> dict

using ``dnspython`` for DNS and ``python-whois`` for registration data, with a
hard 3s timeout and try/except fallbacks. On any failure a feature takes its
documented default and ``network_features_available`` becomes ``False``.

Two categories of tests live here:

1. **Fallback / graceful-degradation tests** (always meaningful). We force the
   underlying libraries to fail and assert the documented defaults. Any
   reasonable implementation that wraps its calls in try/except passes these
   regardless of internal structure — this is the blueprint's core promise
   (FLAG 4: "a failed WHOIS lookup should not fail the entire analysis").

2. **Positive-path tests** (best effort). We inject fake DNS/WHOIS results by
   monkeypatching the third-party libraries at their canonical locations. If
   the injection does not take effect for a given implementation (detected via
   ``network_features_available``), the positive assertions skip rather than
   producing a misleading failure.

No test here performs a real DNS or WHOIS lookup.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from conftest import attr_or_skip, import_or_skip, make_dns_resolver, make_whois_record

# The 18 network feature names, in blueprint order, with their documented
# defaults when the lookup is unavailable.
NETWORK_DEFAULTS = {
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
NETWORK_FEATURE_NAMES = list(NETWORK_DEFAULTS.keys())


@pytest.fixture
def extract():
    module = import_or_skip("features.network")
    return attr_or_skip(module, "extract_network_features"), module


def _force_all_lookups_to_fail(monkeypatch, network_module):
    """Patch every plausible DNS/WHOIS entry point to raise.

    This guarantees the implementation's except branches run, whichever call
    style it uses.
    """

    def _boom(*args, **kwargs):
        raise RuntimeError("network disabled for test")

    # dnspython canonical entry points.
    try:
        import dns.resolver  # type: ignore

        monkeypatch.setattr(dns.resolver, "resolve", _boom, raising=False)
        monkeypatch.setattr(dns.resolver.Resolver, "resolve", _boom, raising=False)
    except Exception:
        pass

    # python-whois canonical entry point.
    try:
        import whois  # type: ignore

        monkeypatch.setattr(whois, "whois", _boom, raising=False)
    except Exception:
        pass

    # Names possibly re-imported into the feature module's namespace.
    for attr in ("whois", "dns"):
        if hasattr(network_module, attr):
            monkeypatch.setattr(network_module, attr, _make_failing_namespace(), raising=False)


def _make_failing_namespace():
    class _Failing:
        def __getattr__(self, _name):
            def _boom(*a, **k):
                raise RuntimeError("network disabled for test")

            return _boom

    return _Failing()


def _inject_working_lookups(monkeypatch, network_module, dns_map, ttl, whois_record):
    """Best-effort injection of successful DNS + WHOIS responses."""
    resolver = make_dns_resolver(dns_map, ttl=ttl)

    try:
        import dns.resolver  # type: ignore

        monkeypatch.setattr(dns.resolver, "resolve", resolver, raising=False)
        monkeypatch.setattr(
            dns.resolver.Resolver, "resolve",
            lambda self, name, rdtype="A", *a, **k: resolver(name, rdtype),
            raising=False,
        )
    except Exception:
        pass

    try:
        import whois  # type: ignore

        monkeypatch.setattr(whois, "whois", lambda *a, **k: whois_record, raising=False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Shape / completeness.
# ---------------------------------------------------------------------------
def test_returns_all_eighteen_network_features(extract, monkeypatch):
    fn, module = extract
    _force_all_lookups_to_fail(monkeypatch, module)
    feats = fn("example.com")
    for name in NETWORK_FEATURE_NAMES:
        assert name in feats, f"missing network feature: {name}"


# ---------------------------------------------------------------------------
# Fallback behaviour (FLAG 4). These are the load-bearing guarantees.
# ---------------------------------------------------------------------------
def test_all_defaults_when_lookups_fail(extract, monkeypatch):
    fn, module = extract
    _force_all_lookups_to_fail(monkeypatch, module)
    feats = fn("some-unreachable-domain-xyz.test")

    assert bool(feats["network_features_available"]) is False
    assert bool(feats["dns_resolves"]) is False
    for name, default in NETWORK_DEFAULTS.items():
        assert feats[name] == default, (
            f"{name} should fall back to {default!r} on lookup failure, "
            f"got {feats[name]!r}"
        )


def test_failure_does_not_raise(extract, monkeypatch):
    """A failed lookup must never propagate an exception to the caller."""
    fn, module = extract
    _force_all_lookups_to_fail(monkeypatch, module)
    # Should return a dict, not blow up.
    assert isinstance(fn("anything.example"), dict)


# ---------------------------------------------------------------------------
# Positive DNS path (best effort — skips if mocks can't be injected).
# ---------------------------------------------------------------------------
def test_dns_records_detected(extract, monkeypatch):
    fn, module = extract
    dns_map = {
        "A": ["1.2.3.4", "1.2.3.5"],
        "MX": ["mail.example.com"],
        "NS": ["ns1.example.com", "ns2.example.com"],
        "TXT": ["v=spf1 -all"],
        "AAAA": ["2606:2800:220:1:248:1893:25c8:1946"],
    }
    _inject_working_lookups(
        monkeypatch, module, dns_map, ttl=3600,
        whois_record=make_whois_record(
            creation_date=datetime.now() - timedelta(days=1000),
            expiration_date=datetime.now() + timedelta(days=365),
        ),
    )
    feats = fn("example.com")
    if not feats.get("network_features_available"):
        pytest.skip("DNS/WHOIS mock injection did not take for this implementation.")

    assert bool(feats["has_a_record"]) is True
    assert feats["num_a_records"] == 2
    assert bool(feats["has_mx_record"]) is True
    assert bool(feats["has_ns_record"]) is True
    assert feats["num_ns_records"] == 2
    assert bool(feats["has_txt_record"]) is True
    assert bool(feats["dns_resolves"]) is True


def test_fast_flux_flag(extract, monkeypatch):
    """is_fast_flux := dns_ttl < 300 AND num_a_records > 3 (blueprint #45)."""
    fn, module = extract
    dns_map = {
        "A": ["1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4", "5.5.5.5"],
        "NS": ["ns1.example.com"],
    }
    _inject_working_lookups(
        monkeypatch, module, dns_map, ttl=60,
        whois_record=make_whois_record(),
    )
    feats = fn("fastflux.example")
    if not feats.get("network_features_available"):
        pytest.skip("DNS mock injection did not take for this implementation.")

    assert feats["num_a_records"] > 3
    assert feats["dns_ttl"] < 300
    assert bool(feats["is_fast_flux"]) is True


# ---------------------------------------------------------------------------
# Positive WHOIS path (best effort).
# ---------------------------------------------------------------------------
def test_newly_registered_flag(extract, monkeypatch):
    """is_newly_registered should be True for a very young domain (< 30 days)."""
    fn, module = extract
    created = datetime.now() - timedelta(days=4)
    expires = datetime.now() + timedelta(days=361)
    _inject_working_lookups(
        monkeypatch, module,
        dns_map={"A": ["1.2.3.4"], "NS": ["ns1.example.com"]},
        ttl=3600,
        whois_record=make_whois_record(creation_date=created, expiration_date=expires),
    )
    feats = fn("brand-new-domain.example")
    if not feats.get("network_features_available"):
        pytest.skip("WHOIS mock injection did not take for this implementation.")

    assert 0 <= feats["domain_age_days"] <= 10
    assert bool(feats["is_newly_registered"]) is True
    # registration_length_days ~= expiration - creation (about a year here)
    assert feats["registration_length_days"] > 300


def test_established_domain_not_newly_registered(extract, monkeypatch):
    fn, module = extract
    created = datetime.now() - timedelta(days=3650)  # ~10 years old
    expires = datetime.now() + timedelta(days=365)
    _inject_working_lookups(
        monkeypatch, module,
        dns_map={"A": ["1.2.3.4"], "NS": ["ns1.example.com"]},
        ttl=3600,
        whois_record=make_whois_record(creation_date=created, expiration_date=expires),
    )
    feats = fn("old-domain.example")
    if not feats.get("network_features_available"):
        pytest.skip("WHOIS mock injection did not take for this implementation.")

    assert feats["domain_age_days"] > 3000
    assert bool(feats["is_newly_registered"]) is False


def test_whois_privacy_detected_from_raw_text(extract, monkeypatch):
    fn, module = extract
    record = make_whois_record(
        creation_date=datetime.now() - timedelta(days=500),
        expiration_date=datetime.now() + timedelta(days=200),
        registrar="NameCheap",
        country=None,
        raw_text="Registrant is REDACTED FOR PRIVACY by whoisguard.",
    )
    _inject_working_lookups(
        monkeypatch, module,
        dns_map={"A": ["1.2.3.4"], "NS": ["ns1.example.com"]},
        ttl=3600, whois_record=record,
    )
    feats = fn("hidden-owner.example")
    if not feats.get("network_features_available"):
        pytest.skip("WHOIS mock injection did not take for this implementation.")

    assert bool(feats["whois_privacy_enabled"]) is True
