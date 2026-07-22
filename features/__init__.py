"""
features package
================

Feature extraction and assembly for ADIS (AI-Powered Domain Intelligence System).

This package turns a raw domain string into the fixed, ordered 48-element
feature vector consumed by the LightGBM model. It is composed of four modules:

    constants.py    Canonical feature ordering, defaults, and categorical metadata.
    structural.py   The 30 structural / lexical features (pure string analysis).
    network.py      The 18 network (DNS + WHOIS) features (live lookups).
    assembler.py    Merges structural + network into the ordered numpy vector.

Public API
----------
Everything most callers need is re-exported at the package level, so downstream
code (the ``/analyze`` route, the training feature builder, tests) can simply::

    from features import assemble_feature_vector, FEATURE_NAMES

Vector assembly (``features.assembler``):
    assemble_feature_vector, assemble_feature_dict, assemble_batch,
    vector_to_dict, feature_names, FeatureExtractionError, DEFAULT_NETWORK_TIMEOUT

Feature metadata (``features.constants``):
    FEATURE_NAMES, FEATURE_COUNT, STRUCTURAL_FEATURE_NAMES, NETWORK_FEATURE_NAMES,
    CATEGORICAL_FEATURES, BOOLEAN_FEATURES, NETWORK_FEATURE_DEFAULTS, CATEGORICAL_UNKNOWN

Structural extraction (``features.structural`` — imported eagerly):
    extract_structural_features, structural_feature_values,
    normalize_domain, split_domain

Network extraction (``features.network`` — imported lazily):
    extract_network_features, default_network_features

Import strategy
---------------
``structural.py`` has only optional third-party dependencies (``tldextract`` and
``python-Levenshtein``, each with pure-Python fallbacks), so it is imported
eagerly and its helpers are available immediately.

``network.py`` hard-imports ``dnspython``, ``python-whois`` and ``tldextract``.
To keep ``import features`` cheap and dependency-light for consumers that only
need structural features or metadata (e.g. the training preprocessor), the two
network callables are resolved lazily via :pep:`562` module ``__getattr__``.
They import ``features.network`` on first access; if the DNS/WHOIS libraries
are not installed, a clear ``FeatureExtractionError`` names the missing
dependencies rather than surfacing a bare ``ImportError``.

Ordering integrity
-------------------
``structural.py`` and ``network.py`` each re-declare their feature-name lists.
The blueprint warns that this order must never drift from ``constants.py`` or
previously trained models become invalid. This package therefore cross-checks
the structural names at import time, and the network names/defaults the first
time the network module is loaded, raising immediately on any mismatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# Eager, dependency-safe re-exports: constants + assembler.
# ---------------------------------------------------------------------------
from .constants import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORICAL_UNKNOWN,
    FEATURE_COUNT,
    FEATURE_NAMES,
    NETWORK_FEATURE_DEFAULTS,
    NETWORK_FEATURE_NAMES,
    STRUCTURAL_FEATURE_NAMES,
)
from .assembler import (
    DEFAULT_NETWORK_TIMEOUT,
    FeatureExtractionError,
    assemble_batch,
    assemble_feature_dict,
    assemble_feature_vector,
    feature_names,
    vector_to_dict,
)

# ---------------------------------------------------------------------------
# Eager structural import (safe: only optional third-party deps, all with
# graceful fallbacks inside structural.py).
# ---------------------------------------------------------------------------
from .structural import (
    STRUCTURAL_FEATURE_NAMES as _STRUCTURAL_NAMES_IMPL,
    extract_structural_features,
    normalize_domain,
    split_domain,
    structural_feature_values,
)

__version__ = "0.2.0"

# ---------------------------------------------------------------------------
# Integrity check: the structural extractor's declared order must match the
# canonical order in constants.py. Fail loudly and immediately on drift.
# ---------------------------------------------------------------------------
if tuple(_STRUCTURAL_NAMES_IMPL) != tuple(STRUCTURAL_FEATURE_NAMES):
    raise FeatureExtractionError(
        "feature order mismatch: features/structural.py STRUCTURAL_FEATURE_NAMES "
        "does not match features/constants.py. These must be identical or the "
        "trained model's feature columns will be misaligned. Reconcile the two "
        "lists before continuing."
    )

# ---------------------------------------------------------------------------
# Lazy network exposure (PEP 562). The network module pulls in DNS/WHOIS
# libraries, so it is only imported when one of these names is first accessed.
# ---------------------------------------------------------------------------
_NETWORK_EXPORTS: frozenset[str] = frozenset(
    {"extract_network_features", "default_network_features"}
)
_NETWORK_DEPS = "dnspython, python-whois, tldextract"
_network_verified = False


def _verify_network_consistency(network_module) -> None:
    """Assert network.py's declared names/defaults match constants.py."""
    if tuple(network_module.NETWORK_FEATURE_NAMES) != tuple(NETWORK_FEATURE_NAMES):
        raise FeatureExtractionError(
            "feature order mismatch: features/network.py NETWORK_FEATURE_NAMES "
            "does not match features/constants.py. Reconcile the two lists "
            "before continuing (model columns would otherwise misalign)."
        )
    if dict(network_module.NETWORK_FEATURE_DEFAULTS) != dict(NETWORK_FEATURE_DEFAULTS):
        raise FeatureExtractionError(
            "default mismatch: features/network.py NETWORK_FEATURE_DEFAULTS does "
            "not match features/constants.py. The same fallback values must be "
            "used at training and inference time."
        )


def _load_network():
    """Import features.network on demand, verifying ordering exactly once."""
    global _network_verified
    try:
        from . import network as _network_module
    except ImportError as exc:
        raise FeatureExtractionError(
            "features/network.py could not be imported. It requires the DNS/WHOIS "
            f"libraries ({_NETWORK_DEPS}); install them (see requirements.txt) to "
            "use network feature extraction. Structural features and the assembler "
            "work without them."
        ) from exc
    if not _network_verified:
        _verify_network_consistency(_network_module)
        _network_verified = True
    return _network_module


def __getattr__(name: str):
    """Resolve the lazily-exposed network callables on first access (PEP 562)."""
    if name in _NETWORK_EXPORTS:
        module = _load_network()
        try:
            return getattr(module, name)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise FeatureExtractionError(
                f"features/network.py does not define {name!r}."
            ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily-exposed network names in dir()/tab-completion."""
    return sorted(set(globals()) | _NETWORK_EXPORTS)


# Let static analysers / IDEs see the lazy names without importing at runtime.
if TYPE_CHECKING:  # pragma: no cover
    from .network import default_network_features, extract_network_features


__all__ = [
    # version
    "__version__",
    # assembly (features.assembler)
    "assemble_feature_vector",
    "assemble_feature_dict",
    "assemble_batch",
    "vector_to_dict",
    "feature_names",
    "FeatureExtractionError",
    "DEFAULT_NETWORK_TIMEOUT",
    # metadata (features.constants)
    "FEATURE_NAMES",
    "FEATURE_COUNT",
    "STRUCTURAL_FEATURE_NAMES",
    "NETWORK_FEATURE_NAMES",
    "CATEGORICAL_FEATURES",
    "BOOLEAN_FEATURES",
    "NETWORK_FEATURE_DEFAULTS",
    "CATEGORICAL_UNKNOWN",
    # structural extraction (features.structural, eager)
    "extract_structural_features",
    "structural_feature_values",
    "normalize_domain",
    "split_domain",
    # network extraction (features.network, lazy)
    "extract_network_features",
    "default_network_features",
]
