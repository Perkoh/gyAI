"""
ml/training/train.py
=====================
gyAI — AI-Powered Domain Intelligence System (ADIS)

Train the production LightGBM domain classifier (blueprint build order step 10).

Pipeline
--------
    feature matrix (feature_builder.py)
        -> preprocess.py            (encode, stratified split, SMOTE, save encoders)
        -> stratified K-fold CV      (this module: estimate performance + rounds)
        -> final fit on training set (this module)
        -> evaluate on held-out test (this module: honest, real-distribution metrics)
        -> gate check + save model   (ml/models/lgbm_model_v<version>.pkl)

Targets (blueprint Phase 3)
---------------------------
    AUC-ROC > 0.95,  F1 > 0.90,  FPR < 2%
These are checked and reported. Use ``--strict`` to make an unmet target a
non-zero exit (e.g. to fail CI), otherwise they are warnings.

Class imbalance (blueprint FLAG 8)
----------------------------------
LightGBM is trained with ``is_unbalance=True`` by default. When preprocess has
already SMOTE-resampled the training set this can mildly over-correct; disable
with ``--no-unbalance`` if you rely purely on resampling.

Model file (blueprint FLAG 9)
-----------------------------
The fitted scikit-learn ``LGBMClassifier`` is pickled to
``ml/models/lgbm_model_v<version>.pkl`` so the Flask API can load it at startup,
alongside a ``.meta.json`` sidecar carrying the metrics/params that populate the
Supabase ``model_versions`` registry. The model is fit on the feature-name
DataFrame so it retains column names in canonical order for SHAP (explainer.py).

Usage
-----
    python -m ml.training.train                       # defaults
    python -m ml.training.train --input ml/data/processed/features.csv \
        --version 1.0.0 --cv-folds 5 --strict

    from ml.training.train import train
    result = train("ml/data/processed/features.csv", version="1.0.0")
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

try:  # joblib ships with scikit-learn; used to persist the model.
    import joblib
    _HAVE_JOBLIB = True
except Exception:  # pragma: no cover
    import pickle
    _HAVE_JOBLIB = False

# LightGBM is required to *train*, but importing this module (for its metric
# helpers, or in tooling) must not fail if it is absent.
try:
    import lightgbm as lgb
    _HAVE_LIGHTGBM = True
except Exception:
    lgb = None  # type: ignore
    _HAVE_LIGHTGBM = False

from features import FEATURE_COUNT, FEATURE_NAMES
from ml.training.preprocess import PreprocessResult, preprocess

# Structured logging (loguru per blueprint §8.1) with a stdlib fallback.
try:  # pragma: no cover
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    class _BraceLogger:
        def __init__(self, name): self._log = logging.getLogger(name)
        def _emit(self, lvl, m, a, k):
            if self._log.isEnabledFor(lvl):
                try: text = m.format(*a, **k) if (a or k) else m
                except Exception: text = m
                self._log.log(lvl, text)
        def debug(self, m, *a, **k): self._emit(logging.DEBUG, m, a, k)
        def info(self, m, *a, **k): self._emit(logging.INFO, m, a, k)
        def warning(self, m, *a, **k): self._emit(logging.WARNING, m, a, k)
        def error(self, m, *a, **k): self._emit(logging.ERROR, m, a, k)

    logger = _BraceLogger("adis.train")


DEFAULT_VERSION = "1.0.0"
DEFAULT_INPUT = Path("ml/data/processed/features.csv")
DEFAULT_MODELS_DIR = Path("ml/models")

# Blueprint Phase 3 acceptance targets.
TARGET_AUC_ROC = 0.95
TARGET_F1 = 0.90
TARGET_FPR = 0.02
DECISION_THRESHOLD = 0.50  # probability cut for class assignment / F1 / FPR


# ===========================================================================
# Metrics & target gate (pure sklearn — unit-testable without LightGBM)
# ===========================================================================
def compute_metrics(
    y_true, y_proba, threshold: float = DECISION_THRESHOLD
) -> dict[str, Any]:
    """Compute the metrics ADIS cares about from probabilities.

    Returns accuracy, precision, recall, f1, auc_roc, false-positive-rate and
    the confusion-matrix cells. AUC uses probabilities; the rest use the
    thresholded prediction.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    # AUC needs both classes present; guard degenerate folds.
    try:
        auc = float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": auc,
        "fpr": fpr,
        "threshold": float(threshold),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
    }


def check_targets(
    metrics: dict[str, Any],
    *,
    target_auc: float = TARGET_AUC_ROC,
    target_f1: float = TARGET_F1,
    target_fpr: float = TARGET_FPR,
) -> tuple[bool, list[str]]:
    """Compare metrics to the blueprint targets. Returns (passed, failures)."""
    failures: list[str] = []
    auc = metrics.get("auc_roc", float("nan"))
    if not (auc > target_auc):
        failures.append(f"AUC-ROC {auc:.4f} <= target {target_auc}")
    if not (metrics.get("f1", 0.0) > target_f1):
        failures.append(f"F1 {metrics.get('f1', 0.0):.4f} <= target {target_f1}")
    if not (metrics.get("fpr", 1.0) < target_fpr):
        failures.append(f"FPR {metrics.get('fpr', 1.0):.4f} >= target {target_fpr}")
    return (not failures), failures


# ===========================================================================
# LightGBM parameters
# ===========================================================================
def build_lgbm_params(
    *,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    n_estimators: int = 1000,
    is_unbalance: bool = True,
    random_state: int = 42,
) -> dict[str, Any]:
    """Sensible LightGBM defaults for imbalanced binary domain classification."""
    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": n_estimators,       # upper bound; early stopping trims it
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "is_unbalance": is_unbalance,       # blueprint FLAG 8
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
        "metric": "auc",
    }


def _require_lightgbm() -> None:
    if not _HAVE_LIGHTGBM:
        raise RuntimeError(
            "lightgbm is not installed. Install it (see requirements.txt: "
            "lightgbm 4.x) to train the model."
        )


# ===========================================================================
# Cross-validation
# ===========================================================================
def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any],
    categorical_features: list[str],
    *,
    cv_folds: int = 5,
    threshold: float = DECISION_THRESHOLD,
    random_state: int = 42,
    early_stopping_rounds: int = 50,
) -> dict[str, Any]:
    """Stratified K-fold CV; returns aggregate metrics and mean best-iteration.

    Fold count is clamped to the minority-class size so every fold keeps both
    classes. With fewer than two usable folds CV is skipped and the caller
    falls back to the configured ``n_estimators``.
    """
    _require_lightgbm()
    min_class = int(pd.Series(y).value_counts().min())
    folds = max(2, min(cv_folds, min_class))
    if min_class < 2:
        logger.warning("minority class < 2; skipping cross-validation")
        return {"skipped": True, "folds": 0, "best_iteration": int(params["n_estimators"])}

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    aucs: list[float] = []
    f1s: list[float] = []
    fprs: list[float] = []
    best_iters: list[int] = []

    for i, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            categorical_feature=categorical_features or "auto",
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        best_iter = getattr(model, "best_iteration_", None) or int(params["n_estimators"])
        best_iters.append(int(best_iter))

        proba = model.predict_proba(X_va)[:, 1]
        m = compute_metrics(y_va, proba, threshold)
        aucs.append(m["auc_roc"]); f1s.append(m["f1"]); fprs.append(m["fpr"])
        logger.info(
            "  fold {}/{}: AUC={:.4f} F1={:.4f} FPR={:.4f} (best_iter={})",
            i, folds, m["auc_roc"], m["f1"], m["fpr"], best_iter,
        )

    summary = {
        "skipped": False,
        "folds": folds,
        "auc_mean": float(np.nanmean(aucs)), "auc_std": float(np.nanstd(aucs)),
        "f1_mean": float(mean(f1s)), "f1_std": float(pstdev(f1s)) if len(f1s) > 1 else 0.0,
        "fpr_mean": float(mean(fprs)), "fpr_std": float(pstdev(fprs)) if len(fprs) > 1 else 0.0,
        "best_iteration": int(round(mean(best_iters))),
        "best_iterations": best_iters,
    }
    logger.info(
        "CV ({} folds): AUC={:.4f}±{:.4f} F1={:.4f}±{:.4f} FPR={:.4f}±{:.4f} | rounds≈{}",
        folds, summary["auc_mean"], summary["auc_std"], summary["f1_mean"],
        summary["f1_std"], summary["fpr_mean"], summary["fpr_std"], summary["best_iteration"],
    )
    return summary


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any],
    categorical_features: list[str],
    n_estimators: int,
):
    """Fit the final model on the full training set with a fixed round count."""
    _require_lightgbm()
    final_params = {**params, "n_estimators": int(max(1, n_estimators))}
    model = lgb.LGBMClassifier(**final_params)
    model.fit(X, y, categorical_feature=categorical_features or "auto")
    logger.info("final model trained on {} rows, {} trees", len(X), final_params["n_estimators"])
    return model


# ===========================================================================
# Persistence
# ===========================================================================
def save_model(
    model,
    metrics: dict[str, Any],
    *,
    version: str,
    models_dir: "str | Path",
    training_samples: int,
    cv_summary: dict[str, Any],
    params: dict[str, Any],
    categorical_features: list[str],
    encoder_path: Optional[str],
    resampled: bool,
    notes: str = "",
) -> tuple[Path, Path]:
    """Persist the model (.pkl) plus a metrics/metadata sidecar (.meta.json)."""
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"lgbm_model_v{version}.pkl"
    meta_path = out_dir / f"lgbm_model_v{version}.meta.json"

    if _HAVE_JOBLIB:
        joblib.dump(model, model_path)
    else:  # pragma: no cover
        with open(model_path, "wb") as fh:
            pickle.dump(model, fh)

    # Keys aligned with the Supabase model_versions registry (blueprint §9).
    meta = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1"],
        "auc_roc": metrics["auc_roc"],
        "precision_score": metrics["precision"],
        "recall_score": metrics["recall"],
        "false_positive_rate": metrics["fpr"],
        "training_samples": int(training_samples),
        "feature_count": FEATURE_COUNT,
        "feature_names": list(FEATURE_NAMES),
        "categorical_features": list(categorical_features),
        "decision_threshold": metrics["threshold"],
        "confusion_matrix": metrics["confusion"],
        "resampled": bool(resampled),
        "cv_summary": cv_summary,
        "lgbm_params": params,
        "encoder_path": str(encoder_path) if encoder_path else None,
        "notes": notes,
        "is_production": False,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("saved model -> {}", model_path)
    logger.info("saved metadata -> {}", meta_path)
    return model_path, meta_path


# ===========================================================================
# Orchestration
# ===========================================================================
@dataclass
class TrainResult:
    model: Any
    version: str
    test_metrics: dict[str, Any]
    cv_summary: dict[str, Any]
    targets_passed: bool
    target_failures: list[str]
    model_path: Optional[Path]
    meta_path: Optional[Path]
    params: dict[str, Any] = field(default_factory=dict)


def train(
    source: "str | Path | pd.DataFrame" = DEFAULT_INPUT,
    *,
    version: str = DEFAULT_VERSION,
    models_dir: "str | Path" = DEFAULT_MODELS_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    n_estimators: int = 1000,
    is_unbalance: bool = True,
    resample: str = "auto",
    scale: bool = False,
    threshold: float = DECISION_THRESHOLD,
    early_stopping_rounds: int = 50,
    notes: str = "",
    save: bool = True,
    pre: Optional[PreprocessResult] = None,
) -> TrainResult:
    """Full training run: preprocess -> CV -> final fit -> evaluate -> save."""
    _require_lightgbm()

    # 1) Preprocess (encode, split, SMOTE, save encoders) — reuse an existing
    #    PreprocessResult if the caller already produced one.
    if pre is None:
        pre = preprocess(
            source, test_size=test_size, random_state=random_state,
            resample=resample, scale=scale, models_dir=models_dir,
        )
    logger.info(
        "training on {} rows ({} features), evaluating on {} held-out rows",
        len(pre.X_train), FEATURE_COUNT, len(pre.X_test),
    )

    params = build_lgbm_params(
        learning_rate=learning_rate, num_leaves=num_leaves,
        n_estimators=n_estimators, is_unbalance=is_unbalance, random_state=random_state,
    )

    # 2) Cross-validate to estimate performance and the number of boosting rounds.
    cv_summary = cross_validate_model(
        pre.X_train, pre.y_train, params, pre.categorical_features,
        cv_folds=cv_folds, threshold=threshold, random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
    )
    final_rounds = cv_summary.get("best_iteration", n_estimators)

    # 3) Fit final model on the full training set.
    model = train_final_model(
        pre.X_train, pre.y_train, params, pre.categorical_features, final_rounds
    )

    # 4) Evaluate on the untouched test set (true class distribution).
    proba_test = model.predict_proba(pre.X_test)[:, 1]
    test_metrics = compute_metrics(pre.y_test, proba_test, threshold)
    logger.info(
        "TEST: acc={:.4f} precision={:.4f} recall={:.4f} F1={:.4f} AUC={:.4f} FPR={:.4f}",
        test_metrics["accuracy"], test_metrics["precision"], test_metrics["recall"],
        test_metrics["f1"], test_metrics["auc_roc"], test_metrics["fpr"],
    )

    passed, failures = check_targets(test_metrics)
    if passed:
        logger.info("✓ all targets met (AUC>{}, F1>{}, FPR<{})", TARGET_AUC_ROC, TARGET_F1, TARGET_FPR)
    else:
        for f in failures:
            logger.warning("target not met: {}", f)

    # 5) Persist.
    model_path = meta_path = None
    if save:
        model_path, meta_path = save_model(
            model, test_metrics, version=version, models_dir=models_dir,
            training_samples=len(pre.X_train), cv_summary=cv_summary, params=params,
            categorical_features=pre.categorical_features,
            encoder_path=str(pre.encoder_path) if pre.encoder_path else None,
            resampled=pre.resampled, notes=notes,
        )

    return TrainResult(
        model=model, version=version, test_metrics=test_metrics, cv_summary=cv_summary,
        targets_passed=passed, target_failures=failures,
        model_path=model_path, meta_path=meta_path, params=params,
    )


# ===========================================================================
# CLI
# ===========================================================================
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="train", description="Train the ADIS LightGBM domain classifier.",
    )
    p.add_argument("--input", "-i", default=str(DEFAULT_INPUT),
                   help=f"feature matrix csv/parquet (default: {DEFAULT_INPUT})")
    p.add_argument("--models-dir", "-m", default=str(DEFAULT_MODELS_DIR),
                   help=f"output dir for model + encoders (default: {DEFAULT_MODELS_DIR})")
    p.add_argument("--version", "-v", default=DEFAULT_VERSION,
                   help=f"model version tag (default: {DEFAULT_VERSION})")
    p.add_argument("--test-size", type=float, default=0.2, help="test fraction (default: 0.2)")
    p.add_argument("--random-state", type=int, default=42, help="RNG seed (default: 42)")
    p.add_argument("--cv-folds", type=int, default=5, help="stratified CV folds (default: 5)")
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--n-estimators", type=int, default=1000, help="max boosting rounds (early stopping trims)")
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--no-unbalance", action="store_true", help="disable LightGBM is_unbalance=True")
    p.add_argument("--resample", choices=("auto", "smote", "none"), default="auto",
                   help="passed to preprocess (default: auto)")
    p.add_argument("--scale", action="store_true", help="scale continuous features (not needed for trees)")
    p.add_argument("--threshold", type=float, default=DECISION_THRESHOLD)
    p.add_argument("--notes", default="", help="free-text note stored in metadata")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero if any performance target is not met")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if not _HAVE_LIGHTGBM:
        logger.error("lightgbm is not installed; cannot train (see requirements.txt)")
        return 1
    try:
        result = train(
            args.input, version=args.version, models_dir=args.models_dir,
            test_size=args.test_size, random_state=args.random_state, cv_folds=args.cv_folds,
            learning_rate=args.learning_rate, num_leaves=args.num_leaves,
            n_estimators=args.n_estimators, is_unbalance=not args.no_unbalance,
            resample=args.resample, scale=args.scale, threshold=args.threshold,
            early_stopping_rounds=args.early_stopping_rounds, notes=args.notes,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("training failed: {}", exc)
        return 1

    print(json.dumps({
        "version": result.version,
        "test_metrics": result.test_metrics,
        "targets_passed": result.targets_passed,
        "target_failures": result.target_failures,
        "model_path": str(result.model_path) if result.model_path else None,
    }, indent=2))

    if args.strict and not result.targets_passed:
        logger.error("strict mode: performance targets not met")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())