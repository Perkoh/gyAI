"""
ml/training/finetune.py
=======================
gyAI — AI-Powered Domain Intelligence System (ADIS)

Continue training ("finetune") the structural-only phase-1 model with network
features enabled, per the two-phase plan:

    Phase 1  feature_builder --no-network  ->  train.py  -> lgbm_model_v1.0.0.pkl
             (48 columns; the 18 network columns sit at constant defaults, so
              LightGBM never splits on them — the model is structural-only)

    Phase 2  feature_builder (network ON, live corpus)   -> THIS MODULE
             loads v1.0.0, adds new boosting rounds that CAN split on the now-
             varying network columns, evaluates, saves lgbm_model_v<version>.pkl

How the saved model is accessed
-------------------------------
``train.py::save_model`` pickles the fitted sklearn ``LGBMClassifier`` with
``joblib.dump`` to ``ml/models/lgbm_model_v1.0.0.pkl``. Therefore:

    model_v1   = joblib.load("ml/models/lgbm_model_v1.0.0.pkl")  # LGBMClassifier
    booster_v1 = model_v1.booster_                               # the trees

The finetune hook is LightGBM's ``init_model`` argument to ``fit``:

    lgb.LGBMClassifier(**params).fit(X2, y2, init_model=booster_v1, ...)

``n_estimators`` in ``params`` then means *additional* trees stacked on top of
the frozen phase-1 trees, fitted to the phase-1 model's residuals. This is
genuine gradient-boosting finetuning. (``train.py`` never passes ``init_model``
— that is why the capability is not visible anywhere in the existing pipeline.)

Encoder policy (the part that silently breaks if done wrong)
------------------------------------------------------------
``preprocess.preprocess()`` ALWAYS refits encoders and overwrites
``label_encoder.pkl`` — never call it here. Phase-1 trees split on the integer
codes produced by the phase-1 encoders, so those codes are load-bearing:

* ``tld``            REUSED from the phase-1 bundle verbatim. It was fully
                     real in phase 1 (structural feature); refitting on the
                     phase-2 subset would re-number the codes and corrupt
                     every phase-1 split on ``tld`` without any error.
* ``whois_country``  REFIT on the phase-2 training split. In the structural-
                     only matrix it was "unknown" everywhere, so the phase-1
                     encoder knows exactly one class; reusing it would map
                     every real country back to "unknown" and the network
                     finetune could never use the feature. Refitting is SAFE
                     precisely because the column was constant in phase 1 —
                     no phase-1 tree can split on a constant, so nothing
                     depends on its old coding.

The rule is applied generically: an encoder is refit only when the phase-1
encoder is degenerate (n_classes <= 1); otherwise it is reused. The merged
bundle is saved back to ``label_encoder.pkl`` (after backing up the phase-1
bundle) so production inference encodes exactly like this finetuned model.

Usage
-----
    python -m ml.training.finetune \
        --input ml/data/processed/features_full.parquet \
        --phase1-model ml/models/lgbm_model_v1.0.0.pkl \
        --version 1.1.0

    from ml.training.finetune import finetune
    result = finetune("ml/data/processed/features_full.parquet",
                      phase1_model_path="ml/models/lgbm_model_v1.0.0.pkl",
                      version="1.1.0")
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import joblib
    _HAVE_JOBLIB = True
except Exception:  # pragma: no cover
    import pickle
    _HAVE_JOBLIB = False

try:
    import lightgbm as lgb
    _HAVE_LIGHTGBM = True
except Exception:  # pragma: no cover
    lgb = None  # type: ignore
    _HAVE_LIGHTGBM = False

from features import CATEGORICAL_FEATURES, FEATURE_COUNT, FEATURE_NAMES
from ml.training.preprocess import (
    DEFAULT_MODELS_DIR,
    ENCODER_FILENAME,
    CategoricalEncoder,
    apply_categorical_encoders,
    load_encoder_bundle,
    load_feature_matrix,
    resample_training_set,
    save_encoder_bundle,
    split_features_labels,
)
from ml.training.train import (
    DECISION_THRESHOLD,
    build_lgbm_params,
    check_targets,
    compute_metrics,
    save_model,
)

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

    logger = _BraceLogger("adis.finetune")


DEFAULT_PHASE1_MODEL = Path("ml/models/lgbm_model_v1.0.0.pkl")
DEFAULT_VERSION = "1.1.0"
#: Fraction of the training split held out to early-stop the NEW trees.
VALIDATION_FRACTION = 0.10


def _require_lightgbm() -> None:
    if not _HAVE_LIGHTGBM:
        raise RuntimeError(
            "lightgbm is not installed. Install it (see requirements.txt: "
            "lightgbm 4.x) to finetune the model."
        )


def _load_pickle(path: Path) -> Any:
    if _HAVE_JOBLIB:
        return joblib.load(path)
    with open(path, "rb") as fh:  # pragma: no cover
        return pickle.load(fh)


# ===========================================================================
# Phase-1 artefact loading + integrity guards
# ===========================================================================
def load_phase1_model(path: "str | Path"):
    """Load the phase-1 model and return ``(model, booster)``.

    ``train.py`` saves the sklearn ``LGBMClassifier``; ``.booster_`` holds the
    trees. A raw ``Booster`` (if someone saved one directly) also works —
    ``init_model`` accepts either.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"phase-1 model not found at {p}. Run phase-1 training first "
            f"(python -m ml.training.train on the --no-network matrix)."
        )
    model = _load_pickle(p)
    booster = getattr(model, "booster_", model)

    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and int(n_in) != FEATURE_COUNT:
        raise ValueError(
            f"phase-1 model expects {n_in} features but the pipeline defines "
            f"{FEATURE_COUNT}. The finetune requires the SAME 48-column vector "
            f"in both phases (structural-only phase 1 must be built with "
            f"--no-network, which defaults the network columns, not drops them)."
        )
    logger.info("loaded phase-1 model from {} ({} features)", p, n_in or "?")
    return model, booster


def load_phase1_encoders(path: "str | Path") -> dict[str, Any]:
    """Load and sanity-check the phase-1 encoder bundle."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"phase-1 encoder bundle not found at {p}. It is written by "
            f"preprocess during phase-1 training (label_encoder.pkl)."
        )
    bundle = load_encoder_bundle(p)

    if list(bundle.get("feature_names", [])) != list(FEATURE_NAMES):
        raise ValueError(
            "feature_names in the phase-1 encoder bundle do not match the "
            "current features package — the feature order has drifted since "
            "phase 1. Finetuning across a changed feature order is invalid; "
            "retrain phase 1 instead."
        )
    if bundle.get("scaler") is not None:
        raise ValueError(
            "the phase-1 pipeline used a fitted scaler. finetune.py assumes "
            "the default unscaled pipeline (scaling is unnecessary for "
            "LightGBM); re-run phase 1 without --scale."
        )
    return bundle


def build_finetune_encoders(
    phase1_encoders: dict[str, Any],
    X_train: pd.DataFrame,
    categorical_features: list[str],
) -> tuple[dict[str, Any], list[str]]:
    """Merge phase-1 encoders with phase-2 refits where safe.

    Reuse an encoder whenever it carries real categories (its integer codes are
    load-bearing for phase-1 trees). Refit ONLY when the phase-1 encoder is
    degenerate (n_classes <= 1, i.e. the column was constant in phase 1 — the
    ``whois_country`` case), which is exactly when re-coding cannot affect any
    phase-1 split.

    Returns ``(encoders, refit_names)``.
    """
    merged: dict[str, Any] = {}
    refit: list[str] = []
    for name in categorical_features:
        enc = phase1_encoders.get(name)
        n_classes = getattr(enc, "n_classes", None)
        if enc is None:
            raise ValueError(
                f"phase-1 bundle has no encoder for categorical feature "
                f"{name!r}; the bundle is incomplete or from a different "
                f"pipeline version."
            )
        # CategoricalEncoder always contains the reserved "unknown" class, so
        # a phase-1 constant column yields exactly 1 class.
        if n_classes is not None and n_classes <= 1:
            new_enc = CategoricalEncoder().fit(X_train[name])
            merged[name] = new_enc
            refit.append(name)
            logger.info(
                "encoder {!r}: phase-1 was constant (n_classes={}) -> REFIT "
                "on phase-2 train ({} classes). Safe: no phase-1 tree can "
                "split on a constant column.",
                name, n_classes, new_enc.n_classes,
            )
        else:
            merged[name] = enc
            logger.info(
                "encoder {!r}: REUSED from phase 1 ({} classes) — its codes "
                "are load-bearing for phase-1 tree splits.",
                name, n_classes,
            )
    return merged, refit


# ===========================================================================
# Orchestration
# ===========================================================================
@dataclass
class FinetuneResult:
    model: Any
    version: str
    test_metrics: dict[str, Any]
    phase1_test_metrics: Optional[dict[str, Any]]
    targets_passed: bool
    target_failures: list[str]
    model_path: Optional[Path]
    meta_path: Optional[Path]
    encoder_path: Optional[Path]
    refit_encoders: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


def finetune(
    source: "str | Path | pd.DataFrame",
    *,
    phase1_model_path: "str | Path" = DEFAULT_PHASE1_MODEL,
    phase1_encoder_path: "str | Path | None" = None,
    version: str = DEFAULT_VERSION,
    models_dir: "str | Path" = DEFAULT_MODELS_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
    learning_rate: float = 0.02,
    num_leaves: int = 31,
    new_estimators: int = 500,
    early_stopping_rounds: int = 50,
    is_unbalance: bool = True,
    resample: str = "auto",
    threshold: float = DECISION_THRESHOLD,
    notes: str = "",
    save: bool = True,
) -> FinetuneResult:
    """Finetune the phase-1 model on a network-enabled feature matrix.

    ``source`` must be a matrix built by feature_builder WITH network features
    (i.e. WITHOUT --no-network), ideally on the live-domain corpus.
    """
    _require_lightgbm()
    models_dir = Path(models_dir)
    if phase1_encoder_path is None:
        phase1_encoder_path = models_dir / ENCODER_FILENAME

    # 1) Phase-1 artefacts -------------------------------------------------
    model_v1, booster_v1 = load_phase1_model(phase1_model_path)
    bundle_v1 = load_phase1_encoders(phase1_encoder_path)

    # 2) Phase-2 data ------------------------------------------------------
    df = load_feature_matrix(source)
    X, y = split_features_labels(df)
    if y.nunique() < 2:
        raise ValueError("phase-2 dataset must contain both classes (0 and 1)")

    net_avail = pd.to_numeric(
        X.get("network_features_available", pd.Series(0, index=X.index)),
        errors="coerce",
    ).fillna(0)
    if float(net_avail.mean()) < 0.5:
        logger.warning(
            "only {:.0%} of phase-2 rows have network_features_available=1 — "
            "is this really the network-enabled matrix (built WITHOUT "
            "--no-network) on the live corpus?",
            float(net_avail.mean()),
        )

    categorical_features = [f for f in CATEGORICAL_FEATURES if f in X.columns]
    categorical_indices = [list(FEATURE_NAMES).index(f) for f in categorical_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("phase-2 split: {} train / {} test (stratified)", len(X_train), len(X_test))

    # 3) Encoders: reuse tld, refit degenerate (whois_country) -------------
    encoders, refit_names = build_finetune_encoders(
        bundle_v1["encoders"], X_train, categorical_features
    )
    X_train = apply_categorical_encoders(X_train, encoders)
    X_test = apply_categorical_encoders(X_test, encoders)

    # 4) Internal validation split for early-stopping the NEW trees, then
    #    resample ONLY the fit portion (never the validation slice). --------
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=VALIDATION_FRACTION,
        random_state=random_state, stratify=y_train,
    )
    X_fit, y_fit, resampled = resample_training_set(
        X_fit, y_fit,
        categorical_indices=categorical_indices,
        mode=resample,
        random_state=random_state,
    )

    # 5) Continue training from the phase-1 booster ------------------------
    params = build_lgbm_params(
        learning_rate=learning_rate, num_leaves=num_leaves,
        n_estimators=new_estimators,        # = ADDITIONAL trees on top of v1
        is_unbalance=is_unbalance, random_state=random_state,
    )
    model_v2 = lgb.LGBMClassifier(**params)
    model_v2.fit(
        X_fit, y_fit,
        init_model=booster_v1,              # ← THE finetune hook
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=categorical_features or "auto",
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    logger.info(
        "finetune fit complete: {} phase-2 rows, up to {} new trees on top of "
        "the phase-1 model (resampled={})",
        len(X_fit), new_estimators, resampled,
    )

    # 6) Evaluate — and compare with phase-1 on the SAME held-out set ------
    proba_v2 = model_v2.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, proba_v2, threshold)

    phase1_metrics: Optional[dict[str, Any]] = None
    try:
        proba_v1 = model_v1.predict_proba(X_test)[:, 1]
        phase1_metrics = compute_metrics(y_test, proba_v1, threshold)
        logger.info(
            "phase-1 on this test set: AUC={:.4f} F1={:.4f} FPR={:.4f}",
            phase1_metrics["auc_roc"], phase1_metrics["f1"], phase1_metrics["fpr"],
        )
    except Exception as exc:  # raw Booster / API mismatch — comparison optional
        logger.warning("could not score phase-1 model for comparison: {}", exc)

    logger.info(
        "FINETUNED TEST: acc={:.4f} precision={:.4f} recall={:.4f} F1={:.4f} "
        "AUC={:.4f} FPR={:.4f}",
        test_metrics["accuracy"], test_metrics["precision"], test_metrics["recall"],
        test_metrics["f1"], test_metrics["auc_roc"], test_metrics["fpr"],
    )
    if phase1_metrics is not None:
        logger.info(
            "delta vs phase-1: AUC {:+.4f} | F1 {:+.4f} | FPR {:+.4f}",
            test_metrics["auc_roc"] - phase1_metrics["auc_roc"],
            test_metrics["f1"] - phase1_metrics["f1"],
            test_metrics["fpr"] - phase1_metrics["fpr"],
        )

    passed, failures = check_targets(test_metrics)
    for f in failures:
        logger.warning("target not met: {}", f)

    # 7) Persist: model v<version> + merged encoder bundle -----------------
    model_path = meta_path = encoder_path = None
    if save:
        # Back up the phase-1 bundle before overwriting label_encoder.pkl.
        p1 = Path(phase1_encoder_path)
        if p1.is_file():
            backup = p1.with_name(p1.stem + ".phase1.backup.pkl")
            shutil.copy2(p1, backup)
            logger.info("backed up phase-1 encoder bundle -> {}", backup)

        encoder_path = save_encoder_bundle(
            encoders, None, categorical_features, categorical_indices,
            models_dir,
            extra_metadata={
                "finetuned_from": str(phase1_model_path),
                "refit_encoders": refit_names,
                "phase": 2,
            },
        )
        model_path, meta_path = save_model(
            model_v2, test_metrics, version=version, models_dir=models_dir,
            training_samples=len(X_fit), cv_summary={"skipped": True, "mode": "finetune"},
            params=params, categorical_features=categorical_features,
            encoder_path=str(encoder_path), resampled=resampled,
            notes=(notes or f"finetuned from {Path(str(phase1_model_path)).name} "
                            f"with network features; refit encoders: {refit_names}"),
        )

    return FinetuneResult(
        model=model_v2, version=version, test_metrics=test_metrics,
        phase1_test_metrics=phase1_metrics, targets_passed=passed,
        target_failures=failures, model_path=model_path, meta_path=meta_path,
        encoder_path=encoder_path, refit_encoders=refit_names, params=params,
    )


# ===========================================================================
# CLI
# ===========================================================================
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="finetune",
        description="Finetune the structural-only ADIS model with network features.",
    )
    p.add_argument("--input", "-i", required=True,
                   help="network-enabled feature matrix (csv/parquet) from feature_builder")
    p.add_argument("--phase1-model", default=str(DEFAULT_PHASE1_MODEL),
                   help=f"phase-1 model .pkl (default: {DEFAULT_PHASE1_MODEL})")
    p.add_argument("--phase1-encoders", default=None,
                   help="phase-1 label_encoder.pkl (default: <models-dir>/label_encoder.pkl)")
    p.add_argument("--models-dir", "-m", default=str(DEFAULT_MODELS_DIR))
    p.add_argument("--version", "-v", default=DEFAULT_VERSION,
                   help=f"version tag for the finetuned model (default: {DEFAULT_VERSION})")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=0.02,
                   help="LR for the NEW trees (default 0.02, lower than phase 1)")
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--new-estimators", type=int, default=500,
                   help="max ADDITIONAL trees (early stopping trims)")
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--no-unbalance", action="store_true")
    p.add_argument("--resample", choices=("auto", "smote", "none"), default="auto")
    p.add_argument("--threshold", type=float, default=DECISION_THRESHOLD)
    p.add_argument("--notes", default="")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero if performance targets are not met")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if not _HAVE_LIGHTGBM:
        logger.error("lightgbm is not installed; cannot finetune (see requirements.txt)")
        return 1
    try:
        result = finetune(
            args.input,
            phase1_model_path=args.phase1_model,
            phase1_encoder_path=args.phase1_encoders,
            version=args.version, models_dir=args.models_dir,
            test_size=args.test_size, random_state=args.random_state,
            learning_rate=args.learning_rate, num_leaves=args.num_leaves,
            new_estimators=args.new_estimators,
            early_stopping_rounds=args.early_stopping_rounds,
            is_unbalance=not args.no_unbalance, resample=args.resample,
            threshold=args.threshold, notes=args.notes,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("finetune failed: {}", exc)
        return 1

    print(json.dumps({
        "version": result.version,
        "test_metrics": result.test_metrics,
        "phase1_test_metrics": result.phase1_test_metrics,
        "targets_passed": result.targets_passed,
        "target_failures": result.target_failures,
        "refit_encoders": result.refit_encoders,
        "model_path": str(result.model_path) if result.model_path else None,
        "encoder_path": str(result.encoder_path) if result.encoder_path else None,
    }, indent=2))

    if args.strict and not result.targets_passed:
        logger.error("strict mode: performance targets not met")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())