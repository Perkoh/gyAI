"""
gyAI — AI-Powered Domain Intelligence System (ADIS)
ml/training/evaluate.py

Model evaluation for the LightGBM domain classifier.

Responsibilities (per PROJECT_BLUEPRINT.md, Phase 3):
  - Compute and print: accuracy, precision, recall, F1, AUC-ROC, PR-AUC,
    confusion matrix, and false-positive rate (FPR).
  - Compare results against the production release targets
    (AUC-ROC > 0.95, F1 > 0.90, FPR < 2%).
  - Return a metrics dict keyed to the Supabase `model_versions` table so
    `train.py` can persist it directly (accuracy, f1_score, auc_roc,
    precision_score, recall_score, false_positive_rate, feature_count, ...).

Two ways to use this module:

  1. Imported by train.py (primary path):

        from ml.training.evaluate import evaluate_model
        metrics = evaluate_model(
            model, X_test, y_test,
            model_version="v1.0.0",
            n_train=len(X_train),
            output_dir="ml/models",
        )

  2. Standalone against a saved model + held-out test set:

        python -m ml.training.evaluate \
            --model ml/models/lgbm_model_v1.0.0.pkl \
            --test-data ml/data/processed/test.pkl

The classification decision threshold is 0.50 (a domain is predicted
"malicious/phishing" when score >= 0.50), matching the extension's
notification decision tree where >= 0.50 triggers a banner. A threshold
sweep is also reported so the operator can see the precision/FPR trade-off
around the 0.80 "high risk / red alert" boundary.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    from loguru import logger
except Exception:  # pragma: no cover - loguru is a hard dep, but degrade gracefully
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("adis.evaluate")  # type: ignore[assignment]

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Default artefact locations (see blueprint section 13, directory structure).
DEFAULT_MODEL_PATH = "ml/models/lgbm_model_v1.0.0.pkl"
DEFAULT_PROCESSED_DIR = "ml/data/processed"

# Binary decision threshold: >= this score => predicted positive (malicious).
# Matches the extension's "show a banner at >= 0.50" behaviour.
DECISION_THRESHOLD = 0.50

# Notification tiers from the blueprint (used only for the reporting breakdown).
SUSPICIOUS_THRESHOLD = 0.50
MALICIOUS_THRESHOLD = 0.80

# Release targets a v1 model must clear before it ships (blueprint Phase 3).
TARGET_AUC_ROC = 0.95
TARGET_F1 = 0.90
TARGET_FPR = 0.02

# The feature vector is fixed at 48 features (blueprint section 6.3). We try to
# read the authoritative list from features/constants.py; if that module isn't
# importable yet (it's built alongside this one), fall back to the known count.
try:  # pragma: no cover - depends on sibling module existing
    from features.constants import FEATURE_NAMES  # type: ignore

    DEFAULT_FEATURE_COUNT = len(FEATURE_NAMES)
except Exception:
    FEATURE_NAMES = None  # type: ignore[assignment]
    DEFAULT_FEATURE_COUNT = 48


# --------------------------------------------------------------------------- #
# Prediction helpers
# --------------------------------------------------------------------------- #

def predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Return the probability of the positive (malicious) class as a 1-D array,
    regardless of whether `model` is a raw lightgbm.Booster or the sklearn
    LGBMClassifier wrapper.
    """
    # sklearn-style estimator (LGBMClassifier, calibrated wrappers, etc.)
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(float)
        return proba.ravel().astype(float)

    # Raw lightgbm.Booster: predict() already yields P(class=1) for binary.
    proba = np.asarray(model.predict(X)).astype(float)
    if proba.ndim == 2:
        # Multiclass booster output; assume last column is the positive class.
        return proba[:, -1]
    return proba.ravel()


def labels_from_scores(scores: Sequence[float]) -> np.ndarray:
    """
    Map continuous risk scores to the three ADIS tiers used by the extension.
    Returned as string labels: 'safe' | 'suspicious' | 'malicious'.
    Purely for the human-readable reporting breakdown.
    """
    scores = np.asarray(scores, dtype=float)
    out = np.full(scores.shape, "safe", dtype=object)
    out[scores >= SUSPICIOUS_THRESHOLD] = "suspicious"
    out[scores >= MALICIOUS_THRESHOLD] = "malicious"
    return out


# --------------------------------------------------------------------------- #
# Metric computation
# --------------------------------------------------------------------------- #

def compute_metrics(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    threshold: float = DECISION_THRESHOLD,
    feature_count: int = DEFAULT_FEATURE_COUNT,
    n_train: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute the full metric suite from ground-truth labels and predicted
    positive-class probabilities.

    The returned dict uses the exact key names expected by the Supabase
    `model_versions` table (accuracy, f1_score, auc_roc, precision_score,
    recall_score, false_positive_rate, feature_count, training_samples) plus a
    few extra diagnostic fields.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_proba = np.asarray(y_proba, dtype=float).ravel()

    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError(
            f"Length mismatch: {y_true.shape[0]} labels vs "
            f"{y_proba.shape[0]} predictions."
        )
    if y_true.size == 0:
        raise ValueError("Cannot evaluate on an empty test set.")

    y_pred = (y_proba >= threshold).astype(int)

    # Confusion matrix with an explicit label order so tn/fp/fn/tp are stable
    # even if the test set happens to contain a single class.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    # Rates. Guard against divide-by-zero on degenerate splits.
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0  # == recall
    tnr = tn / (tn + fp) if (tn + fp) else 0.0  # specificity

    # AUC-ROC / PR-AUC need both classes present in y_true.
    both_classes = np.unique(y_true).size == 2
    auc_roc = float(roc_auc_score(y_true, y_proba)) if both_classes else float("nan")
    pr_auc = (
        float(average_precision_score(y_true, y_proba))
        if both_classes
        else float("nan")
    )

    metrics: Dict[str, Any] = {
        # --- keys mirrored in the Supabase model_versions schema ---
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": auc_roc,
        "precision_score": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_score": float(recall_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fpr),
        "feature_count": int(feature_count),
        "training_samples": int(n_train) if n_train is not None else None,
        # --- extra diagnostics ---
        "pr_auc": pr_auc,
        "false_negative_rate": float(fnr),
        "true_positive_rate": float(tpr),
        "specificity": float(tnr),
        "decision_threshold": float(threshold),
        "test_samples": int(y_true.size),
        "positives_in_test": int(y_true.sum()),
        "negatives_in_test": int(y_true.size - y_true.sum()),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    return metrics


def threshold_sweep(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    thresholds: Sequence[float] = (0.30, 0.50, 0.70, 0.80, 0.90),
) -> list[Dict[str, float]]:
    """
    Report precision / recall / FPR across several thresholds so the operator
    can see the trade-off around the 0.50 (banner) and 0.80 (red alert) lines.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_proba = np.asarray(y_proba, dtype=float).ravel()

    rows: list[Dict[str, float]] = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (int(v) for v in cm.ravel())
        rows.append(
            {
                "threshold": float(t),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
            }
        )
    return rows


def check_targets(metrics: Dict[str, Any]) -> Tuple[bool, list[str]]:
    """
    Compare metrics to the v1 release targets. Returns (all_passed, lines)
    where `lines` are human-readable PASS/FAIL strings for printing.
    """
    checks = [
        ("AUC-ROC", metrics.get("auc_roc"), TARGET_AUC_ROC, "min"),
        ("F1", metrics.get("f1_score"), TARGET_F1, "min"),
        ("FPR", metrics.get("false_positive_rate"), TARGET_FPR, "max"),
    ]
    lines: list[str] = []
    all_passed = True
    for name, value, target, direction in checks:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            passed = False
            verdict = "N/A "
        elif direction == "min":
            passed = value >= target
            verdict = "PASS" if passed else "FAIL"
        else:  # "max"
            passed = value <= target
            verdict = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed
        op = ">=" if direction == "min" else "<="
        shown = "nan" if value is None else f"{value:.4f}"
        lines.append(f"  [{verdict}] {name:<8} {shown}  (target {op} {target})")
    return all_passed, lines


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def format_confusion_matrix(cm: Dict[str, int]) -> str:
    """Render the confusion matrix as an aligned text block."""
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    width = max(len(str(v)) for v in (tn, fp, fn, tp)) + 2
    header = f"{'':>14}{'Pred: safe':>{width + 8}}{'Pred: malicious':>{width + 10}}"
    row_safe = f"{'Actual: safe':>14}{tn:>{width + 8}}{fp:>{width + 10}}"
    row_mal = f"{'Actual: malicious':>14}{fn:>{width + 8}}{tp:>{width + 10}}"
    return "\n".join([header, row_safe, row_mal])


def print_report(
    metrics: Dict[str, Any],
    sweep: Optional[list[Dict[str, float]]] = None,
    model_version: Optional[str] = None,
    tier_breakdown: Optional[Dict[str, int]] = None,
) -> None:
    """Print a full, human-readable evaluation report to the log."""
    title = "ADIS MODEL EVALUATION"
    if model_version:
        title += f"  —  {model_version}"

    lines = ["", "=" * 68, title, "=" * 68]
    lines.append(
        f"Test samples: {metrics['test_samples']}  "
        f"(malicious={metrics['positives_in_test']}, "
        f"safe={metrics['negatives_in_test']})"
    )
    if metrics.get("training_samples"):
        lines.append(f"Training samples: {metrics['training_samples']}")
    lines.append(f"Features: {metrics['feature_count']}")
    lines.append(f"Decision threshold: {metrics['decision_threshold']:.2f}")
    lines.append("-" * 68)

    def fmt(key: str) -> str:
        v = metrics.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "  n/a"
        return f"{v:.4f}"

    lines.append("Core metrics")
    lines.append(f"  Accuracy   : {fmt('accuracy')}")
    lines.append(f"  Precision  : {fmt('precision_score')}")
    lines.append(f"  Recall     : {fmt('recall_score')}")
    lines.append(f"  F1 score   : {fmt('f1_score')}")
    lines.append(f"  AUC-ROC    : {fmt('auc_roc')}")
    lines.append(f"  PR-AUC     : {fmt('pr_auc')}")
    lines.append(f"  FPR        : {fmt('false_positive_rate')}")
    lines.append(f"  FNR        : {fmt('false_negative_rate')}")
    lines.append(f"  Specificity: {fmt('specificity')}")
    lines.append("-" * 68)

    lines.append("Confusion matrix")
    lines.append(format_confusion_matrix(metrics["confusion_matrix"]))
    lines.append("-" * 68)

    if tier_breakdown:
        lines.append("Predicted tier breakdown (extension notification tiers)")
        for tier in ("safe", "suspicious", "malicious"):
            lines.append(f"  {tier:<11}: {tier_breakdown.get(tier, 0)}")
        lines.append("-" * 68)

    if sweep:
        lines.append("Threshold sweep")
        lines.append(
            f"  {'thr':>5} {'precision':>10} {'recall':>8} {'f1':>8} {'fpr':>8}"
        )
        for r in sweep:
            lines.append(
                f"  {r['threshold']:>5.2f} {r['precision']:>10.4f} "
                f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['fpr']:>8.4f}"
            )
        lines.append("-" * 68)

    passed, target_lines = check_targets(metrics)
    lines.append("Release targets")
    lines.extend(target_lines)
    lines.append("-" * 68)
    lines.append(
        "RESULT: "
        + ("ALL TARGETS MET — model is release-ready."
           if passed
           else "TARGETS NOT MET — do not ship this model as-is.")
    )
    lines.append("=" * 68)
    lines.append("")

    logger.info("\n".join(lines))


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def save_metrics(metrics: Dict[str, Any], output_dir: str, model_version: str) -> Path:
    """Write the metrics dict to `<output_dir>/metrics_<version>.json`."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"metrics_{model_version}.json"
    payload = dict(metrics)
    payload["model_version"] = model_version
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info(f"Saved evaluation metrics to {path}")
    return path


def _maybe_save_confusion_plot(
    cm: Dict[str, int], output_dir: str, model_version: str
) -> Optional[Path]:
    """
    Save a confusion-matrix heatmap PNG if matplotlib is available.
    matplotlib is not a hard dependency (it ships transitively with shap), so
    this silently no-ops if it can't be imported.
    """
    try:
        import matplotlib  # type: ignore[import-not-found]

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception:
        return None

    arr = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks([0, 1], labels=["safe", "malicious"])
    ax.set_yticks([0, 1], labels=["safe", "malicious"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {model_version}")
    thresh = arr.max() / 2 if arr.max() else 0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(arr[i, j]), ha="center", va="center",
                color="white" if arr[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"confusion_matrix_{model_version}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info(f"Saved confusion-matrix plot to {path}")
    return path


# --------------------------------------------------------------------------- #
# Top-level entry point (imported by train.py)
# --------------------------------------------------------------------------- #

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: Sequence[int],
    *,
    model_version: str = "v1.0.0",
    threshold: float = DECISION_THRESHOLD,
    n_train: Optional[int] = None,
    output_dir: Optional[str] = None,
    save_plot: bool = True,
    print_output: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a trained LightGBM classifier on a held-out test set.

    Parameters
    ----------
    model : trained lightgbm.Booster or LGBMClassifier
    X_test : feature matrix, shape (n_samples, 48)
    y_test : binary ground-truth labels (0 = safe, 1 = malicious)
    model_version : version tag used in filenames and the report header
    threshold : decision threshold for the positive class (default 0.50)
    n_train : number of training samples, echoed into the returned dict
    output_dir : if given, metrics JSON (+ optional plot) are written here
    save_plot : write a confusion-matrix PNG when matplotlib is available
    print_output : log the full human-readable report

    Returns
    -------
    dict of metrics, keyed to the Supabase `model_versions` schema.
    """
    X_test = np.asarray(X_test)
    feature_count = X_test.shape[1] if X_test.ndim == 2 else DEFAULT_FEATURE_COUNT

    y_proba = predict_proba(model, X_test)
    metrics = compute_metrics(
        y_test,
        y_proba,
        threshold=threshold,
        feature_count=feature_count,
        n_train=n_train,
    )

    sweep = threshold_sweep(y_test, y_proba)

    # Tier breakdown for the notification-tier view.
    tiers = labels_from_scores(y_proba)
    tier_breakdown = {
        "safe": int(np.sum(tiers == "safe")),
        "suspicious": int(np.sum(tiers == "suspicious")),
        "malicious": int(np.sum(tiers == "malicious")),
    }

    if print_output:
        print_report(
            metrics,
            sweep=sweep,
            model_version=model_version,
            tier_breakdown=tier_breakdown,
        )

    if output_dir:
        save_metrics(metrics, output_dir, model_version)
        if save_plot:
            _maybe_save_confusion_plot(
                metrics["confusion_matrix"], output_dir, model_version
            )

    return metrics


# --------------------------------------------------------------------------- #
# Standalone CLI
# --------------------------------------------------------------------------- #

def _load_model(model_path: str) -> Any:
    """Load a model saved by train.py (joblib). Handles bare model or dict."""
    import joblib

    obj = joblib.load(model_path)
    if isinstance(obj, dict):
        for key in ("model", "booster", "estimator", "clf"):
            if key in obj:
                return obj[key]
        raise ValueError(
            f"Loaded a dict from {model_path} but found no model under keys "
            f"model/booster/estimator/clf. Keys present: {list(obj)}"
        )
    return obj


def _load_test_data(test_data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a processed held-out test set. Supported formats:
      - .npz  with arrays X_test/y_test (or X/y)
      - .pkl / .joblib holding a dict with X_test/y_test (or X/y),
        or a tuple/list (X, y)
      - .csv  with feature columns + a trailing 'label' column
    """
    path = Path(test_data_path)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        keys = set(data.files)
        x_key = "X_test" if "X_test" in keys else ("X" if "X" in keys else None)
        y_key = "y_test" if "y_test" in keys else ("y" if "y" in keys else None)
        if not x_key or not y_key:
            raise ValueError(f"{path} must contain X_test/y_test (or X/y). Got {keys}.")
        return np.asarray(data[x_key]), np.asarray(data[y_key])

    if suffix in (".pkl", ".joblib"):
        import joblib

        obj = joblib.load(path)
        if isinstance(obj, dict):
            x_key = "X_test" if "X_test" in obj else ("X" if "X" in obj else None)
            y_key = "y_test" if "y_test" in obj else ("y" if "y" in obj else None)
            if not x_key or not y_key:
                raise ValueError(
                    f"{path} dict must contain X_test/y_test (or X/y). "
                    f"Keys: {list(obj)}"
                )
            return np.asarray(obj[x_key]), np.asarray(obj[y_key])
        if isinstance(obj, (tuple, list)) and len(obj) == 2:
            return np.asarray(obj[0]), np.asarray(obj[1])
        raise ValueError(f"Unsupported object in {path}: {type(obj)}")

    if suffix in (".csv", ".tsv"):
        import pandas as pd

        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        label_col = "label" if "label" in df.columns else df.columns[-1]
        y = df[label_col].to_numpy()
        X = df.drop(columns=[label_col]).to_numpy()
        return X, y

    raise ValueError(
        f"Unsupported test-data format '{suffix}'. Use .npz, .pkl, .joblib, or .csv."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ADIS LightGBM domain classifier."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_PATH,
        help=f"Path to the trained model .pkl (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--test-data", default=f"{DEFAULT_PROCESSED_DIR}/test.pkl",
        help="Path to the held-out test set (.npz/.pkl/.joblib/.csv). "
             "Must provide X_test/y_test (or X/y).",
    )
    parser.add_argument(
        "--model-version", default=None,
        help="Version tag for the report/filenames. "
             "Defaults to the version parsed from the model filename, else v1.0.0.",
    )
    parser.add_argument(
        "--threshold", type=float, default=DECISION_THRESHOLD,
        help=f"Positive-class decision threshold (default {DECISION_THRESHOLD}).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="If set, write metrics JSON (and a plot) to this directory.",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip writing the confusion-matrix PNG even if --output-dir is set.",
    )
    args = parser.parse_args(argv)

    # Derive a version tag from the model filename if not supplied.
    model_version = args.model_version
    if model_version is None:
        stem = Path(args.model).stem  # e.g. lgbm_model_v1.0.0
        model_version = stem.split("_")[-1] if "_v" in stem else "v1.0.0"

    logger.info(f"Loading model:     {args.model}")
    model = _load_model(args.model)
    logger.info(f"Loading test data: {args.test_data}")
    X_test, y_test = _load_test_data(args.test_data)
    logger.info(f"Test set: {X_test.shape[0]} rows, {X_test.shape[1]} features")

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        model_version=model_version,
        threshold=args.threshold,
        output_dir=args.output_dir,
        save_plot=not args.no_plot,
    )

    passed, _ = check_targets(metrics)
    # Non-zero exit code when the model misses release targets — handy for CI.
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())