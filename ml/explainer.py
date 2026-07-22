"""
gyAI — AI-Powered Domain Intelligence System (ADIS)
ml/explainer.py

Per-prediction explainability for the LightGBM domain classifier.

Per PROJECT_BLUEPRINT.md section 6.4: after the model scores a domain, the
`shap` library computes each feature's contribution to *that specific*
prediction. The features that pushed the score toward "malicious" are ranked,
and the top ones are turned into human-readable strings by ml/reason_mapper.py.

Interface (consumed by ml/model_server.py):

    from ml.explainer import build_explainer
    explainer = build_explainer(model, feature_names=FEATURE_NAMES)
    reasons = explainer.explain(feature_vector, score=0.93, top_k=3)
    # -> ["This domain was registered only 4 days ago", ...]

`feature_vector` is the assembled numeric 48-element vector. Pass an optional
`context` dict (tld, matched_keywords, brand, similar_domain, ...) for the
richest wording; without it, generic fallbacks are used.

Robustness
----------
`shap` is the specified backend, but it is imported lazily and the class falls
back automatically to LightGBM's native per-prediction contributions
(`predict(..., pred_contrib=True)`) if SHAP is unavailable or errors. Any
failure in explanation degrades to an empty reason list — explanations are
best-effort and must never break scoring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from loguru import logger
except Exception:  # pragma: no cover - degrade if loguru absent
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("adis.explainer")  # type: ignore[assignment]

from ml import reason_mapper
from ml.reason_mapper import (
    FEATURE_NAMES as DEFAULT_FEATURE_NAMES,
    NETWORK_FEATURES,
    build_reasons,
)

# Only features whose contribution pushes *toward* malicious by at least this
# fraction of the sample's total positive contribution are considered. Filters
# out negligible noise so we don't reach for weak, misleading reasons.
MIN_CONTRIBUTION_SHARE = 0.01

# How many top-ranked features to hand to the reason mapper. Larger than the
# requested reason count so the mapper can skip features that don't render
# (wrong direction, missing template, deduped category) and still fill top_k.
CANDIDATE_POOL = 12


def _resolve_feature_names(
    explicit: Optional[Sequence[str]], model: Any, n_features: Optional[int]
) -> List[str]:
    if explicit:
        return list(explicit)
    # Prefer the authoritative project list if importable.
    try:  # pragma: no cover - depends on features/constants.py existing
        from features.constants import FEATURE_NAMES as PROJECT_NAMES  # type: ignore

        if PROJECT_NAMES:
            return list(PROJECT_NAMES)
    except Exception:
        pass
    # Try the model's own recorded feature names.
    for attr in ("feature_name", "feature_names_in_"):
        obj = getattr(model, attr, None)
        try:
            names = obj() if callable(obj) else obj
        except Exception:
            names = None
        if names is not None and len(list(names)) == (n_features or len(list(names))):
            return list(names)
    return list(DEFAULT_FEATURE_NAMES)


class Explainer:
    """
    Wraps a trained model with a SHAP TreeExplainer (with a native-contribution
    fallback) and renders per-prediction reasons.
    """

    def __init__(
        self,
        model: Any,
        feature_names: Optional[Sequence[str]] = None,
        n_features: Optional[int] = None,
    ) -> None:
        self._model = model
        self._n_features = n_features
        self._feature_names = _resolve_feature_names(feature_names, model, n_features)
        if self._n_features is None:
            self._n_features = len(self._feature_names)
        # Precompute index of the network-availability meta-flag, if present.
        try:
            self._net_flag_idx: Optional[int] = self._feature_names.index(
                "network_features_available"
            )
        except ValueError:
            self._net_flag_idx = None
        self._shap_explainer: Any = None
        self._shap_failed = False
        self._build_shap_explainer()

    # ------------------------------------------------------------------ #
    # Backend construction
    # ------------------------------------------------------------------ #
    def _build_shap_explainer(self) -> None:
        """Build a SHAP TreeExplainer. Silent, lazy — fall back later if absent."""
        try:
            import shap  # lazy: not needed unless we actually explain

            self._shap_explainer = shap.TreeExplainer(self._model)
            logger.info("SHAP TreeExplainer initialised for reason generation.")
        except Exception as exc:
            self._shap_explainer = None
            logger.debug(
                f"SHAP unavailable ({exc}); will use native pred_contrib fallback."
            )

    # ------------------------------------------------------------------ #
    # Contribution computation
    # ------------------------------------------------------------------ #
    def contributions(self, feature_vector: Sequence[float]) -> np.ndarray:
        """
        Return per-feature contributions toward the positive (malicious) class
        for a single sample, as a 1-D array of length n_features. Positive =
        pushed toward malicious.
        """
        X = np.asarray(feature_vector, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        raw = self._shap_contributions(X)
        if raw is None:
            raw = self._native_contributions(X)
        if raw is None:
            return np.zeros(self._n_features, dtype=float)

        row = self._normalize_contribs(raw)
        return row

    def _shap_contributions(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self._shap_explainer is None or self._shap_failed:
            return None
        try:
            # New callable API returns an Explanation with .values.
            try:
                explanation = self._shap_explainer(X)
                vals = np.asarray(explanation.values)
            except Exception:
                vals = self._shap_explainer.shap_values(X)
            return self._pick_positive_class(vals)
        except Exception as exc:
            logger.warning(f"SHAP explanation failed; falling back: {exc}")
            self._shap_failed = True
            return None

    def _native_contributions(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        LightGBM's own per-prediction contributions:
        predict(X, pred_contrib=True) -> shape (n, n_features + 1); the last
        column is the base value, which we drop.
        """
        for target in (self._model, getattr(self._model, "booster_", None)):
            if target is None:
                continue
            try:
                out = np.asarray(target.predict(X, pred_contrib=True), dtype=float)
                return out
            except TypeError:
                continue
            except Exception as exc:
                logger.debug(f"native pred_contrib failed on {target!r}: {exc}")
                continue
        return None

    def _pick_positive_class(self, vals: Any) -> np.ndarray:
        """Reduce SHAP output (list / 2-D / 3-D) to the positive-class array."""
        if isinstance(vals, list):
            # Binary classifier legacy API -> [class0, class1]; take positive.
            vals = vals[-1]
        return np.asarray(vals, dtype=float)

    def _normalize_contribs(self, raw: np.ndarray) -> np.ndarray:
        """
        Coerce whatever the backend returned into a 1-D length-n_features array
        for the (single) sample being explained.
        """
        arr = np.asarray(raw, dtype=float)
        nf = self._n_features

        # Collapse a leading batch dimension to the first sample.
        if arr.ndim == 3:
            # (n, f, c) or (c, n, f)
            if arr.shape[1] == nf:            # (n, f, c) -> positive class
                arr = arr[0, :, -1] if arr.shape[2] > 1 else arr[0, :, 0]
            elif arr.shape[2] == nf:          # (c, n, f) -> positive class, sample 0
                arr = arr[-1, 0, :]
            else:
                arr = arr.reshape(arr.shape[0], -1)[0]
        elif arr.ndim == 2:
            arr = arr[0]
        # else: already 1-D

        # Drop the trailing base-value column from pred_contrib output.
        if arr.shape[0] == nf + 1:
            arr = arr[:nf]
        elif arr.shape[0] != nf:
            arr = arr[:nf] if arr.shape[0] > nf else np.pad(arr, (0, nf - arr.shape[0]))
        return arr

    # ------------------------------------------------------------------ #
    # Public: reason generation
    # ------------------------------------------------------------------ #
    def rank_features(
        self, feature_vector: Sequence[float]
    ) -> List[Tuple[str, float, float]]:
        """
        Return features ranked by contribution toward malicious (descending),
        as (feature_name, feature_value, contribution). Only positive, non-
        negligible contributions are included; network features are dropped
        when the network-availability flag is falsy.
        """
        values = np.asarray(feature_vector, dtype=float).ravel()
        contribs = self.contributions(values)

        network_available = True
        if self._net_flag_idx is not None and self._net_flag_idx < values.shape[0]:
            network_available = values[self._net_flag_idx] >= 0.5

        total_positive = float(np.sum(contribs[contribs > 0])) or 1.0
        order = np.argsort(contribs)[::-1]  # descending

        ranked: List[Tuple[str, float, float]] = []
        for idx in order:
            c = float(contribs[idx])
            if c <= 0 or (c / total_positive) < MIN_CONTRIBUTION_SHARE:
                break  # remaining are smaller/non-incriminating
            name = self._feature_names[idx] if idx < len(self._feature_names) else str(idx)
            if not network_available and name in NETWORK_FEATURES:
                continue
            val = float(values[idx]) if idx < values.shape[0] else 0.0
            ranked.append((name, val, c))
            if len(ranked) >= CANDIDATE_POOL:
                break
        return ranked

    def explain(
        self,
        feature_vector: Sequence[float],
        score: Optional[float] = None,
        top_k: int = 3,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Produce up to `top_k` human-readable reason strings for why the domain
        was flagged. Never raises — returns [] on any internal failure.
        """
        try:
            ranked = self.rank_features(feature_vector)
            candidates = [(name, value) for name, value, _ in ranked]
            return build_reasons(candidates, context=context, top_k=top_k)
        except Exception as exc:
            logger.warning(f"explain() failed; returning no reasons: {exc}")
            return []


# --------------------------------------------------------------------------- #
# Factory (discovered by ModelServer auto-wiring)
# --------------------------------------------------------------------------- #

def build_explainer(
    model: Any,
    feature_names: Optional[Sequence[str]] = None,
    n_features: Optional[int] = None,
) -> Explainer:
    """Construct an Explainer for a trained model."""
    return Explainer(model, feature_names=feature_names, n_features=n_features)
