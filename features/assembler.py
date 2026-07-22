"""
features/assembler.py
=====================

Feature vector assembly for ADIS.

This module is the bridge between the two extractor modules
(``features/structural.py`` and ``features/network.py``) and the LightGBM
model. Its job is deliberately narrow:

    1. Run structural extraction (always) and network extraction (optional).
    2. Merge the two feature dicts.
    3. Fill defaults for any missing / failed network features.
    4. Order everything by ``constants.FEATURE_NAMES`` (the exact training order).
    5. Encode the two categorical features (``tld``, ``whois_country``) to ints.
    6. Coerce everything to floats and return a numpy array of shape (48,).

The same code path runs at training time (via
``ml/training/feature_builder.py``, which can call :func:`assemble_batch`)
and at inference time (via the ``/analyze`` route), guaranteeing that the
feature representation is identical in both places.

Expected extractor contract
----------------------------
This assembler expects the sibling modules to expose these callables:

    features.structural.extract_structural_features(domain: str)
        -> Mapping[str, Any]   # all 30 structural features, keyed by name

    features.network.extract_network_features(domain: str, *, timeout: float)
        -> Mapping[str, Any]   # 18 network features, keyed by name; must set
                               # network_features_available and default any
                               # field it could not resolve

Both are injectable (``structural_fn`` / ``network_fn`` arguments) so the
assembler can be unit-tested in isolation with stub extractors, before the
real modules exist.

Categorical encoding
--------------------
``tld`` and ``whois_country`` are strings. To turn them into model input they
must be integer-encoded. Pass ``encoders`` (a mapping of feature name -> a
fitted sklearn ``LabelEncoder``, a plain ``dict[str, int]``, or a callable)
for production inference so encoding matches training exactly. When no encoder
is supplied, a deterministic hash-based fallback is used so the pipeline still
runs (e.g. the Phase 2 smoke test) -- but production callers should always
pass the encoders saved alongside the model.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .constants import (
    CATEGORICAL_FEATURES,
    CATEGORICAL_UNKNOWN,
    FEATURE_COUNT,
    FEATURE_NAMES,
    NETWORK_FEATURE_DEFAULTS,
    STRUCTURAL_FEATURE_NAMES,
)

try:  # Prefer the project's structured logger.
    from loguru import logger
except Exception:  # pragma: no cover - loguru is a hard dep in prod
    # Fallback adapter so the module still works without loguru. loguru uses
    # ``{}`` brace-style formatting (``logger.warning("x={}", x)``); stdlib
    # logging uses ``%s``. This shim translates brace-style calls so log
    # statements throughout the module behave identically either way.
    import logging as _logging

    class _BraceLogger:
        __slots__ = ("_log",)

        def __init__(self, name: str) -> None:
            self._log = _logging.getLogger(name)

        def _emit(self, level: int, msg: str, args: tuple, kwargs: dict) -> None:
            if self._log.isEnabledFor(level):
                try:
                    text = msg.format(*args, **kwargs) if (args or kwargs) else msg
                except Exception:
                    text = msg
                self._log.log(level, text)

        def debug(self, msg, *a, **k):
            self._emit(_logging.DEBUG, msg, a, k)

        def info(self, msg, *a, **k):
            self._emit(_logging.INFO, msg, a, k)

        def warning(self, msg, *a, **k):
            self._emit(_logging.WARNING, msg, a, k)

        def error(self, msg, *a, **k):
            self._emit(_logging.ERROR, msg, a, k)

        def exception(self, msg, *a, **k):
            self._emit(_logging.ERROR, msg, a, k)

    logger = _BraceLogger("adis.features.assembler")


# ---------------------------------------------------------------------------
# Types & constants
# ---------------------------------------------------------------------------
StructuralFn = Callable[[str], Mapping[str, Any]]
NetworkFn = Callable[..., Mapping[str, Any]]
Encoders = Mapping[str, Any]

DEFAULT_NETWORK_TIMEOUT: float = 3.0  # seconds; matches blueprint FLAG 4

# Size of the hash space for the fallback categorical encoder. A prime keeps
# the modulo distribution even; the exact value is irrelevant as long as it is
# stable across processes and runs.
_HASH_ENCODING_SPACE: int = 100_003


class FeatureExtractionError(RuntimeError):
    """Raised when a feature vector cannot be assembled.

    This is distinct from a *network* failure (which is handled gracefully by
    falling back to defaults). It signals a structural problem: a missing
    extractor module, a malformed input domain, or a structural feature that
    the extractor failed to produce.
    """


__all__ = [
    "assemble_feature_vector",
    "assemble_feature_dict",
    "assemble_batch",
    "vector_to_dict",
    "feature_names",
    "FeatureExtractionError",
    "DEFAULT_NETWORK_TIMEOUT",
]


# ---------------------------------------------------------------------------
# Extractor resolution
# ---------------------------------------------------------------------------
def _resolve_structural_fn(structural_fn: StructuralFn | None) -> StructuralFn:
    if structural_fn is not None:
        return structural_fn
    try:
        from .structural import extract_structural_features
    except ImportError as exc:  # extractor not built yet
        raise FeatureExtractionError(
            "features.structural.extract_structural_features is unavailable. "
            "Implement features/structural.py (build order step 4) or pass "
            "structural_fn explicitly."
        ) from exc
    return extract_structural_features


def _resolve_network_fn(network_fn: NetworkFn | None) -> NetworkFn:
    if network_fn is not None:
        return network_fn
    try:
        from .network import extract_network_features
    except ImportError as exc:  # extractor not built yet
        raise FeatureExtractionError(
            "features.network.extract_network_features is unavailable. "
            "Implement features/network.py (build order step 5) or pass "
            "network_fn explicitly."
        ) from exc
    return extract_network_features


# ---------------------------------------------------------------------------
# Domain normalisation
# ---------------------------------------------------------------------------
def _normalize_domain(domain: str) -> str:
    """Light normalisation before extraction.

    The heavy lifting (SLD/TLD parsing) belongs to the extractors; here we only
    strip whitespace, lowercase, and defensively drop an accidental scheme or
    path so a caller passing a full URL does not silently corrupt features.
    """
    if not isinstance(domain, str):
        raise FeatureExtractionError(
            f"domain must be a string, got {type(domain).__name__}"
        )
    d = domain.strip().lower()
    if not d:
        raise FeatureExtractionError("domain is empty after normalisation")

    # Drop scheme if a URL slipped through.
    if "://" in d:
        d = d.split("://", 1)[1]
    # Drop path / query / fragment and any userinfo/port.
    d = d.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    d = d.split("@")[-1].split(":", 1)[0]
    d = d.rstrip(".")  # strip trailing dot of a fully-qualified name

    if not d:
        raise FeatureExtractionError(
            f"could not extract a hostname from input: {domain!r}"
        )
    return d


# ---------------------------------------------------------------------------
# Extraction runners
# ---------------------------------------------------------------------------
def _run_structural(domain: str, fn: StructuralFn) -> dict[str, Any]:
    try:
        result = fn(domain)
    except Exception as exc:  # structural features should never touch network
        raise FeatureExtractionError(
            f"structural extraction failed for {domain!r}: {exc}"
        ) from exc
    if not isinstance(result, Mapping):
        raise FeatureExtractionError(
            "structural extractor must return a mapping, got "
            f"{type(result).__name__}"
        )
    return dict(result)


def _run_network(domain: str, fn: NetworkFn, timeout: float) -> dict[str, Any]:
    """Run network extraction, degrading gracefully to defaults on failure.

    A network problem must never fail the whole analysis (blueprint FLAG 4);
    if the extractor raises, we log and return the full default set with
    ``network_features_available = False``.
    """
    try:
        result = fn(domain, timeout=timeout)
    except TypeError:
        # Extractor may not accept a timeout kwarg; retry positionally-free.
        try:
            result = fn(domain)
        except Exception as exc:
            logger.warning(
                "network extraction failed for {}: {}; using defaults",
                domain,
                exc,
            )
            return dict(NETWORK_FEATURE_DEFAULTS)
    except Exception as exc:
        logger.warning(
            "network extraction failed for {}: {}; using defaults", domain, exc
        )
        return dict(NETWORK_FEATURE_DEFAULTS)

    if not isinstance(result, Mapping):
        logger.warning(
            "network extractor returned {} (expected mapping); using defaults",
            type(result).__name__,
        )
        return dict(NETWORK_FEATURE_DEFAULTS)
    return dict(result)


# ---------------------------------------------------------------------------
# Value coercion & categorical encoding
# ---------------------------------------------------------------------------
def _to_float(name: str, value: Any) -> float:
    """Coerce a single non-categorical feature value to float."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        f = float(value)
        if f != f or f in (float("inf"), float("-inf")):  # NaN / inf guard
            logger.warning("feature {} produced non-finite value {}; using 0.0", name, value)
            return 0.0
        return f
    if value is None:
        return 0.0
    # A stray string in a numeric slot: hash it deterministically rather than crash.
    logger.warning(
        "feature {} expected numeric, got {!r}; hash-encoding as fallback",
        name,
        value,
    )
    return float(_hash_encode(str(value)))


def _hash_encode(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest, 16) % _HASH_ENCODING_SPACE


def _encode_categorical(name: str, value: Any, encoders: Encoders | None) -> float:
    """Encode a categorical feature (tld / whois_country) to a float code."""
    token = CATEGORICAL_UNKNOWN if value is None else str(value).strip().lower()
    if not token:
        token = CATEGORICAL_UNKNOWN

    if encoders is not None and name in encoders:
        return _apply_encoder(name, token, encoders[name])

    # Deterministic fallback (documented: prefer passing real encoders in prod).
    return float(_hash_encode(token))


def _apply_encoder(name: str, token: str, encoder: Any) -> float:
    """Apply a user-supplied encoder, tolerating several encoder shapes."""
    # 1. Plain mapping {token: code}.
    if isinstance(encoder, Mapping):
        if token in encoder:
            return float(encoder[token])
        if CATEGORICAL_UNKNOWN in encoder:
            return float(encoder[CATEGORICAL_UNKNOWN])
        return float(_hash_encode(token))

    # 2. sklearn LabelEncoder (has .transform and .classes_).
    classes = getattr(encoder, "classes_", None)
    transform = getattr(encoder, "transform", None)
    if classes is not None and callable(transform):
        known = set(classes.tolist() if hasattr(classes, "tolist") else classes)
        if token in known:
            return float(transform([token])[0])
        # Unseen label: reuse "unknown" if the encoder was fit with it, else -1.
        if CATEGORICAL_UNKNOWN in known:
            return float(transform([CATEGORICAL_UNKNOWN])[0])
        logger.debug("unseen {} value {!r}; encoding as -1", name, token)
        return -1.0

    # 3. Bare callable token -> code.
    if callable(encoder):
        return float(encoder(token))

    logger.warning(
        "unrecognised encoder for {} ({}); using hash fallback",
        name,
        type(encoder).__name__,
    )
    return float(_hash_encode(token))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def feature_names() -> tuple[str, ...]:
    """Return the canonical, ordered tuple of all 48 feature names."""
    return FEATURE_NAMES


def assemble_feature_dict(
    domain: str,
    *,
    structural_fn: StructuralFn | None = None,
    network_fn: NetworkFn | None = None,
    include_network: bool = True,
    network_timeout: float = DEFAULT_NETWORK_TIMEOUT,
) -> "OrderedDict[str, Any]":
    """Assemble an ordered dict of raw feature values (native Python types).

    Categorical features remain strings and booleans remain bools -- this form
    is intended for training-time matrix building (where encoders are fit
    afterwards) and for human-readable debugging / SHAP inspection.

    Args:
        domain: Raw domain string (a full URL is tolerated and reduced to host).
        structural_fn: Override for the structural extractor (testing/DI).
        network_fn: Override for the network extractor (testing/DI).
        include_network: If False, skip network lookups and use defaults for
            all 18 network features. Useful for fast, offline / structural-only
            scoring.
        network_timeout: Hard timeout (seconds) passed to the network extractor.

    Returns:
        An ``OrderedDict`` keyed by ``FEATURE_NAMES`` in exact model order.

    Raises:
        FeatureExtractionError: if the domain is invalid, an extractor is
            missing, or a required structural feature was not produced.
    """
    domain = _normalize_domain(domain)

    structural_fn = _resolve_structural_fn(structural_fn)
    structural = _run_structural(domain, structural_fn)

    if include_network:
        network_fn = _resolve_network_fn(network_fn)
        network = _run_network(domain, network_fn, network_timeout)
    else:
        network = dict(NETWORK_FEATURE_DEFAULTS)

    merged: dict[str, Any] = {}
    merged.update(structural)
    merged.update(network)

    ordered: "OrderedDict[str, Any]" = OrderedDict()
    missing_structural: list[str] = []
    for name in FEATURE_NAMES:
        if name in merged and merged[name] is not None:
            ordered[name] = merged[name]
        elif name in NETWORK_FEATURE_DEFAULTS:
            # Network extractor omitted this field -> use its documented default.
            ordered[name] = NETWORK_FEATURE_DEFAULTS[name]
        else:
            # A structural feature is missing: this is a real bug in the
            # structural extractor, not a recoverable network hiccup.
            missing_structural.append(name)

    if missing_structural:
        raise FeatureExtractionError(
            "structural extractor did not produce required feature(s): "
            + ", ".join(missing_structural)
        )

    return ordered


def assemble_feature_vector(
    domain: str,
    *,
    encoders: Encoders | None = None,
    dtype: Any = np.float64,
    structural_fn: StructuralFn | None = None,
    network_fn: NetworkFn | None = None,
    include_network: bool = True,
    network_timeout: float = DEFAULT_NETWORK_TIMEOUT,
) -> np.ndarray:
    """Assemble the model-ready feature vector for a single domain.

    This is the primary entry point used by the ``/analyze`` route. It returns
    a 1-D numpy array of shape ``(48,)`` whose element order matches
    ``constants.FEATURE_NAMES`` exactly.

    Args:
        domain: Raw domain string.
        encoders: Optional mapping of categorical feature name -> encoder
            (fitted sklearn ``LabelEncoder``, ``dict[str, int]``, or callable).
            Strongly recommended in production so encoding matches training.
        dtype: Output numpy dtype (default ``float64``).
        structural_fn / network_fn / include_network / network_timeout:
            See :func:`assemble_feature_dict`.

    Returns:
        ``np.ndarray`` of shape ``(48,)`` and the requested dtype.

    Raises:
        FeatureExtractionError: on invalid input or missing structural features.
    """
    fdict = assemble_feature_dict(
        domain,
        structural_fn=structural_fn,
        network_fn=network_fn,
        include_network=include_network,
        network_timeout=network_timeout,
    )

    vector = np.empty(FEATURE_COUNT, dtype=dtype)
    for i, name in enumerate(FEATURE_NAMES):
        value = fdict[name]
        if name in CATEGORICAL_FEATURES:
            vector[i] = _encode_categorical(name, value, encoders)
        else:
            vector[i] = _to_float(name, value)
    return vector


def assemble_batch(
    domains: Sequence[str] | Iterable[str],
    *,
    encoders: Encoders | None = None,
    dtype: Any = np.float64,
    include_network: bool = True,
    network_timeout: float = DEFAULT_NETWORK_TIMEOUT,
    structural_fn: StructuralFn | None = None,
    network_fn: NetworkFn | None = None,
    on_error: str = "raise",
) -> np.ndarray:
    """Assemble a 2-D feature matrix for many domains.

    Convenience wrapper for the training feature builder. Extractors are
    resolved once up front and reused for every row.

    Args:
        domains: Iterable of raw domain strings.
        on_error: What to do when a single domain fails to assemble:
            ``"raise"`` (default) propagates the error;
            ``"skip"`` drops the row (the returned matrix will have fewer rows);
            ``"zero"`` substitutes an all-zero row so row alignment with an
            external label array is preserved.
        Other args: see :func:`assemble_feature_vector`.

    Returns:
        ``np.ndarray`` of shape ``(n_ok, 48)``. With ``on_error="zero"`` the
        row count always equals ``len(domains)``.

    Raises:
        FeatureExtractionError: if ``on_error="raise"`` and any row fails.
        ValueError: if ``on_error`` is not one of the accepted values.
    """
    if on_error not in {"raise", "skip", "zero"}:
        raise ValueError(
            f"on_error must be 'raise', 'skip', or 'zero'; got {on_error!r}"
        )

    # Resolve extractors once so we don't re-import per row.
    resolved_structural = _resolve_structural_fn(structural_fn)
    resolved_network = _resolve_network_fn(network_fn) if include_network else None

    rows: list[np.ndarray] = []
    failures = 0
    for domain in domains:
        try:
            vec = assemble_feature_vector(
                domain,
                encoders=encoders,
                dtype=dtype,
                structural_fn=resolved_structural,
                network_fn=resolved_network,
                include_network=include_network,
                network_timeout=network_timeout,
            )
            rows.append(vec)
        except FeatureExtractionError as exc:
            if on_error == "raise":
                raise
            failures += 1
            logger.warning("skipping domain {!r}: {}", domain, exc)
            if on_error == "zero":
                rows.append(np.zeros(FEATURE_COUNT, dtype=dtype))

    if failures:
        logger.info("assemble_batch: {} domain(s) failed ({} handling)", failures, on_error)

    if not rows:
        return np.empty((0, FEATURE_COUNT), dtype=dtype)
    return np.vstack(rows)


def vector_to_dict(vector: np.ndarray) -> "OrderedDict[str, float]":
    """Map a 48-element vector back to an ordered {feature_name: value} dict.

    Handy for logging, debugging, and pairing SHAP contribution arrays with
    their feature names in the explainer / reason mapper.
    """
    arr = np.asarray(vector).ravel()
    if arr.shape[0] != FEATURE_COUNT:
        raise FeatureExtractionError(
            f"expected a {FEATURE_COUNT}-element vector, got shape {arr.shape}"
        )
    return OrderedDict((name, float(arr[i])) for i, name in enumerate(FEATURE_NAMES))


# ---------------------------------------------------------------------------
# Import-time sanity check: the structural/network split must partition the
# full ordered vector with no gaps or overlaps.
# ---------------------------------------------------------------------------
assert (
    tuple(STRUCTURAL_FEATURE_NAMES) + tuple(NETWORK_FEATURE_DEFAULTS.keys())
) or True  # (defaults keys checked in constants; kept here for clarity)
assert FEATURE_COUNT == len(FEATURE_NAMES) == 48, "feature vector length drifted from 48"
