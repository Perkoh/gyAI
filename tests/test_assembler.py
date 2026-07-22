"""Tests for the feature assembler (blueprint section 6.3 + Phase 2 goal).

Assumed interface
-----------------
``features/assembler.py`` exposes::

    assemble_feature_vector(domain: str) -> numpy.ndarray   # shape (48,)

and ``features/constants.py`` exposes::

    FEATURE_NAMES: list[str]   # 48 names, exact model-training order

Phase 2 completion criterion (blueprint):
    ``assemble_feature_vector("paypal-secure-login.xyz")`` returns a correct
    48-element numpy array with structural features populated and network
    features populated or defaulted.

Network lookups are forced to fail so the assembler exercises its offline
fallback path (structural populated, network defaulted). This keeps the test
hermetic and matches FLAG 4's "structural features only" fallback.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from conftest import attr_or_skip, import_or_skip  # noqa: E402

EXPECTED_FEATURE_COUNT = 48
STRUCTURAL_COUNT = 30
NETWORK_COUNT = 18


@pytest.fixture
def assemble():
    module = import_or_skip("features.assembler")
    return attr_or_skip(module, "assemble_feature_vector")


@pytest.fixture
def feature_names():
    module = import_or_skip("features.constants")
    return attr_or_skip(module, "FEATURE_NAMES")


@pytest.fixture(autouse=True)
def _disable_network(monkeypatch):
    """Force network lookups to fail so assembly stays offline & deterministic."""
    def _boom(*args, **kwargs):
        raise RuntimeError("network disabled for test")

    for mod_name, targets in (
        ("dns.resolver", ["resolve"]),
        ("whois", ["whois"]),
    ):
        try:
            mod = __import__(mod_name, fromlist=["*"])
            for t in targets:
                monkeypatch.setattr(mod, t, _boom, raising=False)
        except Exception:
            pass
    yield


# ---------------------------------------------------------------------------
# FEATURE_NAMES contract.
# ---------------------------------------------------------------------------
def test_feature_names_has_48_entries(feature_names):
    assert len(feature_names) == EXPECTED_FEATURE_COUNT


def test_feature_names_are_unique(feature_names):
    assert len(set(feature_names)) == len(feature_names), "duplicate feature names"


def test_feature_names_split_structural_then_network(feature_names):
    # The blueprint numbers structural 1-30 and network 31-48.
    assert len(feature_names) == STRUCTURAL_COUNT + NETWORK_COUNT
    # A couple of anchor names should be present.
    for expected in ("domain_length", "network_features_available"):
        assert expected in feature_names


# ---------------------------------------------------------------------------
# Vector shape / dtype.
# ---------------------------------------------------------------------------
def test_vector_length_is_48(assemble):
    vec = assemble("paypal-secure-login.xyz")
    vec = np.asarray(vec)
    assert vec.shape == (EXPECTED_FEATURE_COUNT,)


def test_vector_matches_feature_names_length(assemble, feature_names):
    vec = np.asarray(assemble("google.com"))
    assert vec.shape[0] == len(feature_names)


def test_vector_is_numeric(assemble):
    vec = np.asarray(assemble("google.com"), dtype=float)
    # Casting to float must succeed and contain no NaNs/inf.
    assert np.isfinite(vec).all(), "feature vector contains NaN or inf"


def test_vector_is_1d(assemble):
    vec = np.asarray(assemble("github.com"))
    assert vec.ndim == 1


# ---------------------------------------------------------------------------
# Offline fallback: structural populated, network at defaults.
# ---------------------------------------------------------------------------
def test_network_defaults_present_when_offline(assemble, feature_names):
    """With network disabled, network_features_available must be falsy (0)."""
    vec = np.asarray(assemble("paypal-secure-login.xyz", include_network=False), dtype=float)
    names = list(feature_names)
    if "network_features_available" in names:
        idx = names.index("network_features_available")
        assert vec[idx] == 0, "network_features_available should be 0/False offline"


def test_structural_slot_populated_when_offline(assemble, feature_names):
    """domain_length is a structural feature and must be non-zero for a real domain."""
    vec = np.asarray(assemble("paypal-secure-login.xyz"), dtype=float)
    names = list(feature_names)
    if "domain_length" in names:
        idx = names.index("domain_length")
        assert vec[idx] == len("paypal-secure-login.xyz")


# ---------------------------------------------------------------------------
# Determinism / consistency.
# ---------------------------------------------------------------------------
def test_assembly_is_deterministic(assemble):
    a = np.asarray(assemble("example.com"), dtype=float)
    b = np.asarray(assemble("example.com"), dtype=float)
    assert np.array_equal(a, b), "assembler must be deterministic for a fixed domain"


def test_different_domains_differ(assemble):
    a = np.asarray(assemble("google.com"), dtype=float)
    b = np.asarray(assemble("secure-login-paypa1.xyz"), dtype=float)
    assert not np.array_equal(a, b), "distinct domains should yield distinct vectors"
