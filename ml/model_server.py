"""
gyAI — AI-Powered Domain Intelligence System (ADIS)
ml/model_server.py

Singleton model server: loads the trained LightGBM domain classifier into
memory once, and exposes a single `predict()` interface used by the Flask API.

Design (per PROJECT_BLUEPRINT.md):
  - Section 5.2: "All inference happens inside the Flask API process ... The
    model file loads into memory once at startup."
  - Section 13: model_server.py = "Singleton model loader + predict interface".
  - Phase 3 completion criterion:
        ModelServer().predict(feature_vector)
    must return {score, label, confidence, reasons} for a domain's feature
    vector.
  - Section 10.1: /admin/model/reload must "Reload model from disk without
    restart" -> ModelServer.reload().

Contract of predict():

    server = ModelServer()                 # shared singleton, model already loaded
    result = server.predict(feature_vector)
    # result == {
    #   "score":      0.9341,              # P(malicious), 0..1, rounded to 4dp
    #   "label":      "malicious",         # safe | suspicious | malicious
    #   "confidence": "high",              # low | medium | high
    #   "reasons":    [ "...", "...", ... ] # [] when safe or no explainer wired
    # }

`feature_vector` is the assembled, already-numeric 48-element vector produced by
features/assembler.py (categoricals encoded upstream). This module never touches
DNS/WHOIS or raw domain strings — it is pure model inference.

Reason strings are produced by an *explainer* object (built in ml/explainer.py +
ml/reason_mapper.py, which land right after this file). The server auto-wires an
explainer if that module is importable, and also accepts one via
`attach_explainer()`. If none is available, predictions still succeed with
`reasons: []`.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from loguru import logger
except Exception:  # pragma: no cover - loguru is a hard dep; degrade gracefully
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("adis.model_server")  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Default model artefact (blueprint section 13). Overridable via env / settings.
DEFAULT_MODEL_PATH = "ml/models/lgbm_model_v1.1.0.pkl"
DEFAULT_MODEL_VERSION = "v1.1.0"
DEFAULT_FEATURE_COUNT = 48

# Score -> label tiers (blueprint sections 4.3 / 11.1 / API contract).
SUSPICIOUS_THRESHOLD = 0.50   # >= this and < MALICIOUS_THRESHOLD -> "suspicious"
MALICIOUS_THRESHOLD = 0.80    # >= this -> "malicious"; below SUSPICIOUS -> "safe"

# Confidence tiers. The blueprint fixes the {low, medium, high} vocabulary but
# not the formula, so we derive confidence from how far the score sits from the
# 0.50 decision boundary: certainty = max(score, 1 - score) in [0.5, 1.0].
# This reproduces the contract examples (0.9341 -> high, 0.0213 -> high).
HIGH_CONFIDENCE_CERTAINTY = 0.80
MEDIUM_CONFIDENCE_CERTAINTY = 0.65

# Number of reason strings to surface for a flagged domain (blueprint: top 3).
DEFAULT_TOP_REASONS = 3

_VERSION_RE = re.compile(r"(v\d+\.\d+\.\d+)")


# --------------------------------------------------------------------------- #
# Config resolution (defensive: config/settings.py may not exist yet)
# --------------------------------------------------------------------------- #

def _resolve_model_path(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    try:  # pragma: no cover - depends on sibling module existing
        from config import settings  # type: ignore

        for attr in ("MODEL_PATH", "ADIS_MODEL_PATH"):
            val = getattr(settings, attr, None)
            if val:
                return str(val)
    except Exception:
        pass
    return os.getenv("ADIS_MODEL_PATH") or os.getenv("MODEL_PATH") or DEFAULT_MODEL_PATH


def _resolve_model_version(explicit: Optional[str], model_path: str) -> str:
    if explicit:
        return explicit
    try:  # pragma: no cover
        from config import settings  # type: ignore

        val = getattr(settings, "MODEL_VERSION", None)
        if val:
            return str(val)
    except Exception:
        pass
    env = os.getenv("ADIS_MODEL_VERSION") or os.getenv("MODEL_VERSION")
    if env:
        return env
    # Parse from filename, e.g. lgbm_model_v1.0.0.pkl -> v1.0.0
    m = _VERSION_RE.search(Path(model_path).stem)
    return m.group(1) if m else DEFAULT_MODEL_VERSION


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

class ModelNotLoadedError(RuntimeError):
    """Raised when predict() is called before a model is available."""


class ModelServer:
    """
    Thread-safe singleton around the trained LightGBM classifier.

    Constructing `ModelServer()` always returns the same process-wide instance.
    The first construction determines the model path / version and (by default)
    loads the model eagerly. Subsequent `ModelServer()` calls return the live
    instance and ignore any constructor arguments.

    For tests, pass `auto_load=False` (skips disk I/O) and use
    `reset_model_server()` to clear the singleton between cases.
    """

    _instance: Optional["ModelServer"] = None
    _singleton_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Singleton machinery
    # ------------------------------------------------------------------ #
    def __new__(cls, *args: Any, **kwargs: Any) -> "ModelServer":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_version: Optional[str] = None,
        explainer: Any = None,
        auto_load: bool = True,
        auto_wire_explainer: bool = True,
    ) -> None:
        # Guard against re-initialisation on repeated ModelServer() calls.
        if getattr(self, "_initialized", False):
            if model_path or model_version or explainer is not None:
                logger.debug(
                    "ModelServer already initialised; ignoring new constructor args."
                )
            return

        self._model_path = _resolve_model_path(model_path)
        self._model_version = _resolve_model_version(model_version, self._model_path)
        self._model: Any = None
        self._n_features: int = DEFAULT_FEATURE_COUNT
        self._feature_names: Optional[List[str]] = None
        self._explainer: Any = explainer
        self._auto_wire_explainer = auto_wire_explainer
        self._load_lock = threading.Lock()
        self._initialized = True

        if auto_load:
            self.load()

    # ------------------------------------------------------------------ #
    # Loading / reloading
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """
        Load (or re-load) the model from `self._model_path` into memory.
        Raises FileNotFoundError / ValueError on failure — startup should fail
        loudly if the model artefact is missing or malformed.
        """
        with self._load_lock:
            model, n_features, feature_names = self._load_from_disk(self._model_path)
            self._model = model
            self._n_features = n_features
            self._feature_names = feature_names
            logger.info(
                f"Model loaded: {self._model_path} "
                f"(version={self._model_version}, features={self._n_features})"
            )
            if self._explainer is None and self._auto_wire_explainer:
                self._try_auto_wire_explainer()

    def reload(self) -> Dict[str, Any]:
        """
        Reload the model from disk without restarting the process (backs the
        /admin/model/reload endpoint). Loads the new model first, then swaps the
        reference atomically so concurrent predict() calls never see a half-
        loaded state. Returns the fresh version_info().
        """
        with self._load_lock:
            self._model_version = _resolve_model_version(None, self._model_path)
            model, n_features, feature_names = self._load_from_disk(self._model_path)
            # Atomic swap (CPython reference assignment is atomic).
            self._model = model
            self._n_features = n_features
            self._feature_names = feature_names
            # Explainer wraps the model, so rebuild it against the new one.
            self._explainer = None
            if self._auto_wire_explainer:
                self._try_auto_wire_explainer()
        logger.info(f"Model reloaded from disk (version={self._model_version}).")
        return self.version_info()

    @staticmethod
    def _load_from_disk(model_path: str):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. Train a model first "
                f"(python -m ml.training.train) or set ADIS_MODEL_PATH."
            )
        import joblib

        obj = joblib.load(path)

        # train.py may save either a bare model or a bundle dict.
        feature_names: Optional[List[str]] = None
        model = obj
        if isinstance(obj, dict):
            for key in ("model", "booster", "estimator", "clf"):
                if key in obj:
                    model = obj[key]
                    break
            else:
                raise ValueError(
                    f"Loaded a dict from '{model_path}' with no model under "
                    f"model/booster/estimator/clf. Keys: {list(obj)}"
                )
            fn = obj.get("feature_names")
            if fn:
                feature_names = list(fn)

        n_features = ModelServer._infer_n_features(model, feature_names)
        return model, n_features, feature_names

    @staticmethod
    def _infer_n_features(model: Any, feature_names: Optional[List[str]]) -> int:
        if feature_names:
            return len(feature_names)
        # sklearn wrapper
        n = getattr(model, "n_features_in_", None)
        if isinstance(n, (int, np.integer)):
            return int(n)
        # raw lightgbm.Booster
        try:
            nf = model.num_feature()  # type: ignore[attr-defined]
            if isinstance(nf, (int, np.integer)):
                return int(nf)
        except Exception:
            pass
        return DEFAULT_FEATURE_COUNT

    # ------------------------------------------------------------------ #
    # Explainer wiring
    # ------------------------------------------------------------------ #
    def attach_explainer(self, explainer: Any) -> None:
        """
        Inject a reason-generating explainer. Expected duck-typed interface:

            explainer.explain(feature_vector, score=<float>, top_k=<int>)
                -> list[str]

        The explainer is responsible for the full SHAP -> human-readable-string
        pipeline (ml/explainer.py + ml/reason_mapper.py). Injection is optional;
        without it predictions return reasons: [].
        """
        self._explainer = explainer
        logger.info("Explainer attached to ModelServer.")

    def _try_auto_wire_explainer(self) -> None:
        """Best-effort: build an explainer from ml.explainer if it exists yet."""
        try:  # pragma: no cover - depends on ml/explainer.py existing
            from ml import explainer as explainer_mod  # type: ignore

            builder = None
            for name in ("build_explainer", "create_explainer", "get_explainer"):
                builder = getattr(explainer_mod, name, None)
                if callable(builder):
                    self._explainer = builder(self._model)
                    logger.info(f"Auto-wired explainer via ml.explainer.{name}().")
                    return
            explainer_cls = getattr(explainer_mod, "Explainer", None)
            if callable(explainer_cls):
                self._explainer = explainer_cls(self._model)
                logger.info("Auto-wired explainer via ml.explainer.Explainer().")
        except Exception as exc:
            logger.debug(f"No explainer auto-wired (ok for now): {exc}")

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(
        self,
        feature_vector: Sequence[float],
        explain: bool = True,
        top_reasons: int = DEFAULT_TOP_REASONS,
    ) -> Dict[str, Any]:
        """
        Score a single domain's feature vector.

        Returns {score, label, confidence, reasons}. `reasons` is always []
        for safe domains (per the blueprint's safe-response contract), and is
        populated for suspicious/malicious domains when an explainer is wired
        and `explain=True`.
        """
        x = self._prepare_vector(feature_vector)
        proba = float(self._predict_proba(x)[0])
        return self._assemble_result(proba, x[0], explain=explain, top_reasons=top_reasons)

    def predict_batch(
        self,
        feature_vectors: Sequence[Sequence[float]],
        explain: bool = False,
        top_reasons: int = DEFAULT_TOP_REASONS,
    ) -> List[Dict[str, Any]]:
        """
        Score many feature vectors in one model call (backs /analyze/bulk).
        Reasons default off here to keep bulk latency low.
        """
        X = self._prepare_vector(feature_vectors)
        probas = self._predict_proba(X)
        return [
            self._assemble_result(
                float(probas[i]), X[i], explain=explain, top_reasons=top_reasons
            )
            for i in range(X.shape[0])
        ]

    # ------------------------------------------------------------------ #
    # Internal prediction helpers
    # ------------------------------------------------------------------ #
    def _prepare_vector(self, feature_vector: Sequence[float]) -> np.ndarray:
        if self._model is None:
            raise ModelNotLoadedError(
                "Model is not loaded. Call ModelServer().load() before predict()."
            )
        arr = np.asarray(feature_vector, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(
                f"feature_vector must be 1-D or 2-D, got shape {arr.shape}."
            )

        if arr.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features per sample, "
                f"got {arr.shape[1]}. The assembler must emit the model's "
                f"exact feature order/count."
            )
        if not np.all(np.isfinite(arr)):
            # LightGBM tolerates NaN, but inf usually signals a feature bug.
            n_bad = int(np.sum(~np.isfinite(arr) & ~np.isnan(arr)))
            if n_bad:
                raise ValueError(
                    f"feature_vector contains {n_bad} non-finite (inf) value(s)."
                )
        return arr

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(malicious) as a 1-D array for a Booster or sklearn wrapper."""
        model = self._model
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(X))
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1].astype(float)
            return proba.ravel().astype(float)
        proba = np.asarray(model.predict(X)).astype(float)
        if proba.ndim == 2:
            return proba[:, -1]
        return proba.ravel()

    def _assemble_result(
        self,
        proba: float,
        feature_row: np.ndarray,
        explain: bool,
        top_reasons: int,
    ) -> Dict[str, Any]:
        proba = float(min(max(proba, 0.0), 1.0))
        label = self.score_to_label(proba)
        confidence = self.score_to_confidence(proba)

        reasons: List[str] = []
        if explain and label != "safe":
            reasons = self._generate_reasons(feature_row, proba, top_reasons)

        return {
            "score": round(proba, 4),
            "label": label,
            "confidence": confidence,
            "reasons": reasons,
        }

    def _generate_reasons(
        self, feature_row: np.ndarray, score: float, top_k: int
    ) -> List[str]:
        """
        Best-effort reason generation. A failure here must never break a
        prediction, so all explainer errors are caught and downgraded to [].
        """
        if self._explainer is None:
            return []
        try:
            explain_fn = getattr(self._explainer, "explain", None)
            if not callable(explain_fn):
                return []
            # Try the richest signature first, then progressively simpler ones.
            for call in (
                lambda: explain_fn(feature_row, score=score, top_k=top_k),
                lambda: explain_fn(feature_row, top_k=top_k),
                lambda: explain_fn(feature_row),
            ):
                try:
                    reasons = call()
                    break
                except TypeError:
                    continue
            else:
                return []

            if reasons is None:
                return []
            reasons = [str(r) for r in reasons][:top_k]
            return reasons
        except Exception as exc:
            logger.warning(f"Reason generation failed; returning no reasons: {exc}")
            return []

    # ------------------------------------------------------------------ #
    # Score -> label / confidence (static, reusable, testable)
    # ------------------------------------------------------------------ #
    @staticmethod
    def score_to_label(score: float) -> str:
        if score >= MALICIOUS_THRESHOLD:
            return "malicious"
        if score >= SUSPICIOUS_THRESHOLD:
            return "suspicious"
        return "safe"

    @staticmethod
    def score_to_confidence(score: float) -> str:
        certainty = max(score, 1.0 - score)
        if certainty >= HIGH_CONFIDENCE_CERTAINTY:
            return "high"
        if certainty >= MEDIUM_CONFIDENCE_CERTAINTY:
            return "medium"
        return "low"

    # ------------------------------------------------------------------ #
    # Introspection (for /health, /version, admin)
    # ------------------------------------------------------------------ #
    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def feature_names(self) -> Optional[List[str]]:
        return list(self._feature_names) if self._feature_names else None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def has_explainer(self) -> bool:
        return self._explainer is not None

    def version_info(self) -> Dict[str, Any]:
        """Payload for GET /version."""
        return {
            "model_version": self._model_version,
            "model_path": self._model_path,
            "feature_count": self._n_features,
            "explainer_available": self.has_explainer,
            "loaded": self.is_loaded,
        }

    def health(self) -> Dict[str, Any]:
        """Lightweight readiness payload for GET /health."""
        return {
            "status": "ok" if self.is_loaded else "model_not_loaded",
            "model_loaded": self.is_loaded,
            "model_version": self._model_version,
        }


# --------------------------------------------------------------------------- #
# Module-level accessors
# --------------------------------------------------------------------------- #

def get_model_server(
    model_path: Optional[str] = None,
    model_version: Optional[str] = None,
    auto_load: bool = True,
) -> ModelServer:
    """
    Return the process-wide ModelServer singleton, constructing (and loading)
    it on first call. Call this from the Flask app factory at startup so the
    model is warm before the first request.
    """
    return ModelServer(
        model_path=model_path, model_version=model_version, auto_load=auto_load
    )


def reset_model_server() -> None:
    """Drop the singleton. Primarily for tests; not used in production."""
    with ModelServer._singleton_lock:
        ModelServer._instance = None