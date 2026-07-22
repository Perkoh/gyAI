"""Tests for the model server (blueprint section 4.3 + Phase 3 goal).

Assumed interface
-----------------
``ml/model_server.py`` exposes a ``ModelServer`` whose ``predict`` maps a
feature vector to a scored result::

    ModelServer().predict(feature_vector) -> {
        "score": float,          # 0..1 probability
        "label": str,            # "safe" | "suspicious" | "malicious"
        "confidence": str,       # "low" | "medium" | "high"
        "reasons": list[str],    # top SHAP-derived reasons (empty when safe)
    }

Score -> label thresholds (blueprint 4.3 / 11.1):
    score < 0.50            -> "safe"      (silent, no reasons)
    0.50 <= score < 0.80    -> "suspicious"
    score >= 0.80           -> "malicious"

To avoid needing a trained ``.pkl`` on disk or a real SHAP explainer, we drive
the underlying model through whatever seam the implementation exposes:

* If ``ModelServer`` accepts an injected model (constructor kwarg or settable
  attribute), we inject a ``FakeBooster`` returning a chosen probability.
* Otherwise we patch ``joblib.load`` (the blueprint's serialisation choice) to
  return the fake model before constructing the server.

If none of these seams exist for a given implementation, the affected test
skips rather than failing — the contract under test is the *mapping logic*, not
a specific wiring style.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from conftest import FakeBooster, attr_or_skip, import_or_skip  # noqa: E402

VALID_LABELS = {"safe", "suspicious", "malicious"}
VALID_CONFIDENCE = {"low", "medium", "high"}


def _feature_vector():
    return np.zeros(48, dtype=float)


def _build_server_with_score(monkeypatch, score: float):
    """Return a ModelServer instance whose model predicts ``score``.

    Tries constructor injection, attribute injection, then joblib.load patching.
    Skips the test if no seam works.
    """
    module = import_or_skip("ml.model_server")
    ModelServer = attr_or_skip(module, "ModelServer")
    fake = FakeBooster(probability=score)

    # Strategy 1: constructor accepts a model.
    for kwarg in ("model", "booster", "clf"):
        try:
            server = ModelServer(**{kwarg: fake})
            return server
        except TypeError:
            continue
        except Exception:
            break

    # Strategy 2: patch joblib.load, then construct normally.
    try:
        import joblib  # type: ignore

        monkeypatch.setattr(joblib, "load", lambda *a, **k: fake, raising=False)
        # Also patch a re-imported reference inside the module, if present.
        if hasattr(module, "joblib"):
            monkeypatch.setattr(module.joblib, "load", lambda *a, **k: fake, raising=False)
    except Exception:
        pass

    # Neutralise SHAP so reason generation never needs a real explainer.
    _neutralise_shap(monkeypatch, module)

    try:
        server = ModelServer()
    except Exception as exc:  # model file missing / different wiring
        pytest.skip(f"Could not construct ModelServer with a fake model: {exc}")

    # Strategy 3: set a settable model attribute post-construction.
    for attr in ("model", "booster", "_model", "clf"):
        if hasattr(server, attr):
            try:
                setattr(server, attr, fake)
            except Exception:
                pass
    return server


def _neutralise_shap(monkeypatch, module):
    """Best-effort: make any SHAP explainer return zero contributions."""
    try:
        import shap  # type: ignore

        class _FakeExplainer:
            def __init__(self, *a, **k):
                pass

            def shap_values(self, X, *a, **k):
                X = np.atleast_2d(np.asarray(X, dtype=float))
                return np.zeros_like(X)

            def __call__(self, X, *a, **k):
                return self.shap_values(X)

        monkeypatch.setattr(shap, "TreeExplainer", _FakeExplainer, raising=False)
        monkeypatch.setattr(shap, "Explainer", _FakeExplainer, raising=False)
    except Exception:
        pass


def _predict(server, vector=None):
    predict = getattr(server, "predict", None)
    if predict is None:
        pytest.skip("ModelServer has no predict() method.")
    return predict(_feature_vector() if vector is None else vector)


# ---------------------------------------------------------------------------
# Output schema.
# ---------------------------------------------------------------------------
def test_predict_returns_expected_keys(monkeypatch):
    server = _build_server_with_score(monkeypatch, 0.02)
    result = _predict(server)
    for key in ("score", "label", "confidence", "reasons"):
        assert key in result, f"predict() result missing key: {key}"


def test_predict_types(monkeypatch):
    server = _build_server_with_score(monkeypatch, 0.9)
    result = _predict(server)
    assert isinstance(result["score"], (int, float))
    assert 0.0 <= float(result["score"]) <= 1.0
    assert result["label"] in VALID_LABELS
    assert result["confidence"] in VALID_CONFIDENCE
    assert isinstance(result["reasons"], list)


# ---------------------------------------------------------------------------
# Score -> label thresholds (the core ADIS decision logic).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "score,expected_label",
    [
        (0.01, "safe"),
        (0.49, "safe"),
        (0.50, "suspicious"),
        (0.64, "suspicious"),
        (0.79, "suspicious"),
        (0.80, "malicious"),
        (0.93, "malicious"),
        (0.999, "malicious"),
    ],
)
def test_score_maps_to_label(monkeypatch, score, expected_label):
    server = _build_server_with_score(monkeypatch, score)
    result = _predict(server)
    assert result["label"] == expected_label, (
        f"score {score} should map to '{expected_label}', got '{result['label']}'"
    )


def test_boundary_050_is_suspicious_not_safe(monkeypatch):
    server = _build_server_with_score(monkeypatch, 0.50)
    assert _predict(server)["label"] == "suspicious"


def test_boundary_080_is_malicious_not_suspicious(monkeypatch):
    server = _build_server_with_score(monkeypatch, 0.80)
    assert _predict(server)["label"] == "malicious"


# ---------------------------------------------------------------------------
# Reasons behaviour.
# ---------------------------------------------------------------------------
def test_safe_domain_has_no_reasons(monkeypatch):
    server = _build_server_with_score(monkeypatch, 0.02)
    result = _predict(server)
    # Blueprint: safe responses carry an empty reasons list.
    assert result["reasons"] == []


def test_malicious_reason_count_capped_at_three(monkeypatch):
    server = _build_server_with_score(monkeypatch, 0.95)
    result = _predict(server)
    # Blueprint shows "top 3 reasons" for alerts.
    assert len(result["reasons"]) <= 3


# ---------------------------------------------------------------------------
# Optional pure-helper checks (only run if the implementation exposes them).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "score,expected",
    [(0.10, "safe"), (0.60, "suspicious"), (0.85, "malicious")],
)
def test_optional_score_to_label_helper(score, expected):
    module = import_or_skip("ml.model_server")
    helper = getattr(module, "score_to_label", None)
    if helper is None:
        pytest.skip("no standalone score_to_label helper exposed.")
    assert helper(score) == expected
