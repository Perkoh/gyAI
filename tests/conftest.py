"""Shared pytest fixtures and helpers for the ADIS test suite.

This file centralises everything the individual test modules rely on:

* It puts the project root on ``sys.path`` so the application packages
  (``features``, ``ml``, ``cache``, ``database``, ``api``) import cleanly no
  matter which directory ``pytest`` is invoked from.
* It exposes small "import-or-skip" helpers. ADIS is built test-first: the
  tests describe the contract from ``PROJECT_BLUEPRINT.md`` and are written
  before (or alongside) the implementation. When a module or symbol under test
  does not exist yet, the relevant test *skips* with an explanatory message
  rather than erroring the whole run. As the corresponding component is built,
  its tests light up automatically.
* It provides reusable sample domains and fake DNS/WHOIS/Redis objects so the
  suite never touches the real network, a real Redis, Supabase, or a trained
  model file.

Nothing in here makes live network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

# ---------------------------------------------------------------------------
# Make the project importable (repo root is the parent of tests/).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Canonical sample domains used across the suite (from the blueprint examples).
# ---------------------------------------------------------------------------
SAFE_DOMAINS = ["google.com", "github.com", "amazon.com"]
PHISHING_DOMAINS = [
    "secure-login-paypa1.xyz",
    "amazon-account-verify.com",
    "xn--pypal-4ve.com",
]


# ---------------------------------------------------------------------------
# Resilient import helpers.
# ---------------------------------------------------------------------------
def import_or_skip(module_path: str):
    """Import ``module_path`` or skip the *whole* current test module.

    Thin wrapper over :func:`pytest.importorskip` kept here so intent reads
    clearly at the call sites and the skip reason is uniform.
    """
    return pytest.importorskip(
        module_path,
        reason=f"{module_path} is not implemented yet (test-first placeholder).",
    )


def attr_or_skip(module: Any, name: str) -> Any:
    """Return ``module.name`` or skip the current *test* if it is missing.

    Use inside a test/fixture body (function scope) so a single missing symbol
    skips only the affected test instead of aborting collection.
    """
    obj = getattr(module, name, None)
    if obj is None:
        mod_name = getattr(module, "__name__", str(module))
        pytest.skip(f"{mod_name}.{name} not implemented yet.")
    return obj


def first_attr_or_skip(module: Any, names: list[str]) -> Any:
    """Return the first attribute in ``names`` that exists on ``module``.

    Lets tests tolerate small naming variations in the implementation
    (e.g. ``set_cached_result`` vs ``cache_result``). Skips if none match.
    """
    for name in names:
        obj = getattr(module, name, None)
        if obj is not None:
            return obj
    mod_name = getattr(module, "__name__", str(module))
    pytest.skip(f"{mod_name} exposes none of: {', '.join(names)} (not implemented yet).")


@pytest.fixture
def get_symbol() -> Callable[[Any, str], Any]:
    """Fixture form of :func:`attr_or_skip` for readability in tests."""
    return attr_or_skip


# ---------------------------------------------------------------------------
# Fake DNS answer objects (mimic the dnspython Answer interface just enough).
# ---------------------------------------------------------------------------
class FakeRRSet:
    def __init__(self, ttl: int) -> None:
        self.ttl = ttl


class FakeDNSAnswer:
    """Minimal stand-in for ``dns.resolver.Answer``.

    Supports iteration, ``len()``, and ``.rrset.ttl`` which covers the common
    ways feature code reads DNS results.
    """

    def __init__(self, records: list[str], ttl: int = 3600) -> None:
        self._records = [SimpleNamespace(to_text=lambda r=r: r, address=r) for r in records]
        self.rrset = FakeRRSet(ttl)

    def __iter__(self):
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)


def make_dns_resolver(mapping: dict[str, list[str]], ttl: int = 3600):
    """Build a ``resolve(name, rdtype)`` callable backed by ``mapping``.

    ``mapping`` maps record type (``"A"``, ``"MX"`` ...) to a list of records.
    Missing types raise, mirroring dnspython's ``NoAnswer`` behaviour so the
    feature code's try/except fallbacks are exercised realistically.
    """

    def _resolve(name: str, rdtype: str = "A", *args, **kwargs) -> FakeDNSAnswer:
        key = str(rdtype).upper()
        if key not in mapping:
            raise RuntimeError(f"No {key} record for {name}")
        return FakeDNSAnswer(mapping[key], ttl=ttl)

    return _resolve


@pytest.fixture
def fake_dns_resolver():
    """Factory fixture returning a configurable fake DNS resolve() callable."""
    return make_dns_resolver


# ---------------------------------------------------------------------------
# Fake WHOIS record.
# ---------------------------------------------------------------------------
def make_whois_record(
    creation_date=None,
    expiration_date=None,
    registrar: str | None = "GoDaddy.com, LLC",
    country: str | None = "US",
    raw_text: str = "",
):
    """Return an object shaped like a ``python-whois`` result."""
    return SimpleNamespace(
        creation_date=creation_date,
        expiration_date=expiration_date,
        registrar=registrar,
        country=country,
        text=raw_text,
    )


@pytest.fixture
def fake_whois_record():
    return make_whois_record


# ---------------------------------------------------------------------------
# A fake in-memory Redis client (used when `fakeredis` is unavailable).
# ---------------------------------------------------------------------------
class FakeRedis:
    """Tiny in-memory Redis substitute recording TTLs for assertions."""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.ttls: dict[str, int] = {}

    # basic string ops -----------------------------------------------------
    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None, **kwargs):
        self.store[key] = value
        if ex is not None:
            self.ttls[key] = int(ex)
        return True

    def setex(self, key, ttl, value):
        self.store[key] = value
        self.ttls[key] = int(ttl)
        return True

    def expire(self, key, ttl):
        if key in self.store:
            self.ttls[key] = int(ttl)
            return True
        return False

    def ttl(self, key):
        return self.ttls.get(key, -1)

    def delete(self, *keys):
        removed = 0
        for key in keys:
            if key in self.store:
                del self.store[key]
                self.ttls.pop(key, None)
                removed += 1
        return removed

    def exists(self, key):
        return 1 if key in self.store else 0

    def keys(self, pattern="*"):
        return list(self.store.keys())

    def flushdb(self):
        self.store.clear()
        self.ttls.clear()
        return True

    def scan_iter(self, match=None, **kwargs):
        return iter(list(self.store.keys()))

    def ping(self):
        return True


@pytest.fixture
def fake_redis() -> FakeRedis:
    """A fresh in-memory Redis substitute per test."""
    return FakeRedis()


# ---------------------------------------------------------------------------
# A fake LightGBM booster whose predicted probability is controllable.
# ---------------------------------------------------------------------------
class FakeBooster:
    """Stand-in for a trained LightGBM model.

    ``predict`` returns the configured probability for every input row, which
    lets the model-server tests drive score -> label/confidence mapping
    deterministically without a real ``.pkl`` on disk.
    """

    def __init__(self, probability: float = 0.02) -> None:
        self.probability = probability
        self.num_feature = lambda: 48  # some code calls model.num_feature()

    def predict(self, features, *args, **kwargs):
        try:
            import numpy as np  # local import keeps conftest importable w/o numpy
        except Exception:  # pragma: no cover - numpy is a hard dep in practice
            n = len(features) if hasattr(features, "__len__") else 1
            return [self.probability] * n
        arr = np.atleast_2d(np.asarray(features, dtype=float))
        return np.full(arr.shape[0], self.probability, dtype=float)


@pytest.fixture
def fake_booster():
    """Factory: ``fake_booster(prob)`` -> FakeBooster returning that score."""
    return FakeBooster
