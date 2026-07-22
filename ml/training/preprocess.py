"""
ml/training/preprocess.py
=========================
gyAI — AI-Powered Domain Intelligence System (ADIS)

Turn the raw feature matrix produced by ``feature_builder.py`` into
model-ready training/test splits. Three responsibilities (blueprint build
order step 9 — "Label encoding, scaling, SMOTE if needed"):

  1. **Encode categoricals.** ``tld`` and ``whois_country`` arrive as raw
     strings; they are integer-encoded with a fitted, unseen-safe encoder and
     saved to ``ml/models/label_encoder.pkl`` so the *exact same* mapping is
     reused at inference time by the ``/analyze`` route via the assembler.
  2. **Stratified train/test split.** Class ratios are preserved in both sets
     (blueprint FLAG 8), so evaluation is honest for a rare positive class.
  3. **Handle class imbalance.** When the minority class is severely
     under-represented (worse than the 90:10 guidance in blueprint §7.3), the
     *training set only* is over-sampled with SMOTE. Because two features are
     categorical, ``SMOTENC`` is used (it copies the majority category among
     neighbours instead of interpolating meaningless fractional codes).

Design notes
------------
* Encoders are fit on the **training split only** — never on test — to avoid
  leakage. Categories seen only in the test set (or, later, in production)
  map to a reserved ``"unknown"`` code, mirroring live behaviour exactly.
* Scaling is exposed but **off by default**: LightGBM is tree-based and gains
  nothing from feature scaling. When enabled it touches continuous columns
  only, never the integer categorical codes.
* SMOTE is applied *after* encoding but *before* any scaling, so ``SMOTENC``
  sees the raw integer category codes it needs.
* imbalanced-learn is an optional dependency: if it is not installed,
  resampling is skipped with a warning rather than failing the pipeline.

The saved bundle (``label_encoder.pkl``) is a dict:
    {
      "encoders":  {feature_name: CategoricalEncoder},   # assembler-compatible
      "scaler":    StandardScaler | None,
      "feature_names": [...48...],
      "categorical_features": ["tld", "whois_country"],
      "categorical_indices": [2, 35],
      "metadata":  {...},
    }
At inference:  ``assemble_feature_vector(domain, encoders=bundle["encoders"])``.

Usage
-----
    python -m ml.training.preprocess \
        --input ml/data/processed/features.csv \
        --models-dir ml/models --test-size 0.2

    from ml.training.preprocess import preprocess
    result = preprocess("ml/data/processed/features.csv")
    model.fit(result.X_train, result.y_train,
              categorical_feature=result.categorical_indices)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:  # joblib ships with scikit-learn and pickles sklearn objects efficiently.
    import joblib
    _HAVE_JOBLIB = True
except Exception:  # pragma: no cover
    _HAVE_JOBLIB = False

# imbalanced-learn is optional (blueprint §7.3: SMOTE only "if needed").
try:
    from imblearn.over_sampling import SMOTE, SMOTENC
    _HAVE_IMBLEARN = True
except Exception:
    SMOTE = SMOTENC = None  # type: ignore
    _HAVE_IMBLEARN = False

from features import CATEGORICAL_FEATURES, FEATURE_COUNT, FEATURE_NAMES

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

    logger = _BraceLogger("adis.preprocess")


PREPROCESS_VERSION = "1.0.0"

DEFAULT_INPUT = Path("ml/data/processed/features.csv")
DEFAULT_MODELS_DIR = Path("ml/models")
ENCODER_FILENAME = "label_encoder.pkl"

LABEL_COLUMN = "label"
DOMAIN_COLUMN = "domain"

#: Minority-class fraction below which "auto" resampling kicks in (worse than
#: the 90:10 ratio flagged in blueprint §7.3).
DEFAULT_IMBALANCE_THRESHOLD = 0.10
#: Target minority:majority ratio after SMOTE (≈67:33, within the 70:30–80:20
#: guidance of §7.3 rather than a full 1:1 which can over-correct alongside
#: LightGBM's own ``is_unbalance``).
DEFAULT_SMOTE_RATIO = 0.5


# ===========================================================================
# Categorical encoder — unseen-safe, picklable, assembler-compatible
# ===========================================================================
class CategoricalEncoder:
    """Integer-encode a single categorical string feature.

    Duck-types like a scikit-learn ``LabelEncoder`` (exposes ``classes_`` and
    ``transform``) so it plugs directly into ``features.assembler`` at
    inference time. Unlike ``LabelEncoder`` it is *unseen-safe*: any value not
    present at fit time (including at inference) maps to a reserved
    ``"unknown"`` code instead of raising.
    """

    UNKNOWN = "unknown"

    def __init__(self) -> None:
        self.classes_: np.ndarray = np.array([self.UNKNOWN])
        self._map: dict[str, int] = {self.UNKNOWN: 0}
        self._unknown_code: int = 0

    @staticmethod
    def _norm(value: Any) -> str:
        if value is None:
            return CategoricalEncoder.UNKNOWN
        token = str(value).strip().lower()
        return token if token else CategoricalEncoder.UNKNOWN

    def fit(self, values) -> "CategoricalEncoder":
        cats = sorted({self._norm(v) for v in values})
        if self.UNKNOWN in cats:  # keep UNKNOWN at a stable position (last)
            cats.remove(self.UNKNOWN)
        cats.append(self.UNKNOWN)
        self.classes_ = np.array(cats, dtype=object)
        self._map = {c: i for i, c in enumerate(cats)}
        self._unknown_code = self._map[self.UNKNOWN]
        return self

    def transform(self, values) -> np.ndarray:
        return np.fromiter(
            (self._map.get(self._norm(v), self._unknown_code) for v in values),
            dtype=np.int64,
            count=len(values),
        )

    def fit_transform(self, values) -> np.ndarray:
        return self.fit(values).transform(values)

    @property
    def n_classes(self) -> int:
        return len(self.classes_)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CategoricalEncoder(n_classes={self.n_classes})"


# ===========================================================================
# Result container
# ===========================================================================
@dataclass
class PreprocessResult:
    """Everything ``train.py`` needs to fit and later reproduce the pipeline."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    categorical_features: list[str]
    categorical_indices: list[int]
    encoders: dict[str, CategoricalEncoder]
    scaler: Optional[StandardScaler]
    resampled: bool
    class_balance_before: dict[str, int]
    class_balance_after: dict[str, int]
    encoder_path: Optional[Path] = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# Loading & X/y separation
# ===========================================================================
def load_feature_matrix(source: "str | os.PathLike | pd.DataFrame") -> pd.DataFrame:
    """Load the feature matrix from a path (csv/parquet) or accept a DataFrame."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(
                f"feature matrix not found at {path}. Run feature_builder first "
                f"(python -m ml.training.feature_builder)."
            )
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

    missing = set(FEATURE_NAMES) - set(df.columns)
    if missing:
        raise ValueError(
            f"feature matrix is missing {len(missing)} expected column(s), e.g. "
            f"{sorted(missing)[:5]}. Was it produced by feature_builder?"
        )
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"feature matrix has no '{LABEL_COLUMN}' column")
    return df


def split_features_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X in canonical feature order, y as int), dropping ``domain``."""
    X = df[list(FEATURE_NAMES)].copy()
    y = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
    if y.isna().any():
        raise ValueError("label column contains non-numeric values")
    return X, y.astype(int)


# ===========================================================================
# Categorical encoding
# ===========================================================================
def fit_categorical_encoders(
    X_train: pd.DataFrame, categorical_features: list[str]
) -> dict[str, CategoricalEncoder]:
    """Fit one :class:`CategoricalEncoder` per categorical column (train only)."""
    encoders: dict[str, CategoricalEncoder] = {}
    for feat in categorical_features:
        enc = CategoricalEncoder().fit(X_train[feat].tolist())
        encoders[feat] = enc
        logger.info("encoded '{}': {} categories (+unknown)", feat, enc.n_classes - 1)
    return encoders


def apply_categorical_encoders(
    X: pd.DataFrame, encoders: dict[str, CategoricalEncoder]
) -> pd.DataFrame:
    """Return a copy of ``X`` with categorical columns replaced by int codes."""
    X = X.copy()
    for feat, enc in encoders.items():
        X[feat] = enc.transform(X[feat].tolist())
    return X


# ===========================================================================
# Imbalance handling (SMOTE / SMOTENC)
# ===========================================================================
def _class_counts(y) -> dict[str, int]:
    vc = pd.Series(y).value_counts().to_dict()
    return {"benign": int(vc.get(0, 0)), "malicious": int(vc.get(1, 0))}


def resample_training_set(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    categorical_indices: list[int],
    mode: str = "auto",
    imbalance_threshold: float = DEFAULT_IMBALANCE_THRESHOLD,
    sampling_strategy: "float | str" = DEFAULT_SMOTE_RATIO,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, bool]:
    """Optionally SMOTE-oversample the minority class in the training set.

    Returns ``(X, y, resampled)``. ``mode`` is 'auto' (resample only when the
    minority fraction is below ``imbalance_threshold``), 'smote' (always), or
    'none' (never). SMOTENC is used because the matrix has categorical columns.
    """
    if mode == "none":
        return X_train, y_train, False

    counts = _class_counts(y_train)
    n_min = min(counts["benign"], counts["malicious"])
    n_maj = max(counts["benign"], counts["malicious"])
    total = n_min + n_maj
    minority_fraction = (n_min / total) if total else 0.0

    if mode == "auto" and minority_fraction >= imbalance_threshold:
        logger.info(
            "class balance within tolerance (minority {:.1%} >= {:.0%}); "
            "skipping SMOTE", minority_fraction, imbalance_threshold,
        )
        return X_train, y_train, False

    if not _HAVE_IMBLEARN:
        logger.warning(
            "imbalanced-learn not installed; skipping SMOTE (minority {:.1%}). "
            "Install it (see requirements.txt) or rely on LightGBM is_unbalance=True.",
            minority_fraction,
        )
        return X_train, y_train, False

    if n_min < 2:
        logger.warning("minority class has <2 samples; cannot SMOTE, skipping")
        return X_train, y_train, False

    # SMOTE needs k_neighbors < n_minority; shrink k for tiny minority classes.
    k_neighbors = min(5, n_min - 1)

    try:
        if categorical_indices and SMOTENC is not None:
            sampler = SMOTENC(
                categorical_features=categorical_indices,
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=random_state,
            )
        else:  # pragma: no cover - no-categorical fallback
            sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=random_state,
            )
        X_res, y_res = sampler.fit_resample(X_train, y_train)
    except Exception as exc:  # never let resampling abort training prep
        logger.warning("SMOTE failed ({}); continuing without resampling", exc)
        return X_train, y_train, False

    # fit_resample may return numpy arrays; restore the DataFrame/Series shape.
    if not isinstance(X_res, pd.DataFrame):
        X_res = pd.DataFrame(X_res, columns=list(X_train.columns))
    if not isinstance(y_res, pd.Series):
        y_res = pd.Series(y_res, name=y_train.name)

    logger.info(
        "SMOTE applied (k_neighbors={}): {} -> {} training rows",
        k_neighbors, len(X_train), len(X_res),
    )
    return X_res.reset_index(drop=True), y_res.reset_index(drop=True).astype(int), True


# ===========================================================================
# Optional scaling (continuous columns only; off by default for trees)
# ===========================================================================
def fit_scaler(
    X_train: pd.DataFrame, continuous_features: list[str]
) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train[continuous_features])
    return scaler


def apply_scaler(
    X: pd.DataFrame, scaler: StandardScaler, continuous_features: list[str]
) -> pd.DataFrame:
    X = X.copy()
    X[continuous_features] = scaler.transform(X[continuous_features])
    return X


# ===========================================================================
# Persistence
# ===========================================================================
def save_encoder_bundle(
    encoders: dict[str, CategoricalEncoder],
    scaler: Optional[StandardScaler],
    categorical_features: list[str],
    categorical_indices: list[int],
    models_dir: "str | os.PathLike" = DEFAULT_MODELS_DIR,
    *,
    filename: str = ENCODER_FILENAME,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> Path:
    """Persist encoders (+ optional scaler) to ``ml/models/label_encoder.pkl``."""
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    bundle = {
        "encoders": encoders,
        "scaler": scaler,
        "feature_names": list(FEATURE_NAMES),
        "categorical_features": list(categorical_features),
        "categorical_indices": list(categorical_indices),
        "metadata": {
            "preprocess_version": PREPROCESS_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feature_count": FEATURE_COUNT,
            **(extra_metadata or {}),
        },
    }

    if _HAVE_JOBLIB:
        joblib.dump(bundle, path)
    else:  # pragma: no cover
        with open(path, "wb") as fh:
            pickle.dump(bundle, fh)
    logger.info("saved encoder bundle -> {}", path)
    return path


def load_encoder_bundle(path: "str | os.PathLike") -> dict[str, Any]:
    """Load a saved encoder bundle (for the API / evaluation)."""
    p = Path(path)
    if _HAVE_JOBLIB:
        return joblib.load(p)
    with open(p, "rb") as fh:  # pragma: no cover
        return pickle.load(fh)


# ===========================================================================
# Orchestration
# ===========================================================================
def preprocess(
    source: "str | os.PathLike | pd.DataFrame" = DEFAULT_INPUT,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    resample: str = "auto",
    imbalance_threshold: float = DEFAULT_IMBALANCE_THRESHOLD,
    sampling_strategy: "float | str" = DEFAULT_SMOTE_RATIO,
    scale: bool = False,
    models_dir: "str | os.PathLike" = DEFAULT_MODELS_DIR,
    save_encoders: bool = True,
) -> PreprocessResult:
    """Full preprocessing: load -> split -> encode -> resample -> (scale) -> save.

    Returns a :class:`PreprocessResult` with encoded, split-ready arrays and the
    fitted encoders/scaler. Encoders are persisted to ``label_encoder.pkl`` so
    inference uses an identical mapping.
    """
    if resample not in {"auto", "smote", "none"}:
        raise ValueError(f"resample must be 'auto', 'smote', or 'none'; got {resample!r}")

    df = load_feature_matrix(source)
    X, y = split_features_labels(df)
    balance_before = _class_counts(y)
    logger.info(
        "loaded {} rows | benign={} malicious={}",
        len(X), balance_before["benign"], balance_before["malicious"],
    )
    if y.nunique() < 2:
        raise ValueError("dataset must contain both classes (0 and 1) to train")

    categorical_features = [f for f in CATEGORICAL_FEATURES if f in X.columns]
    categorical_indices = [list(FEATURE_NAMES).index(f) for f in categorical_features]
    continuous_features = [f for f in FEATURE_NAMES if f not in categorical_features]

    # 1) Stratified split (before fitting encoders — no leakage).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("split: {} train / {} test (stratified)", len(X_train), len(X_test))

    # 2) Fit categorical encoders on train, apply to both.
    encoders = fit_categorical_encoders(X_train, categorical_features)
    X_train = apply_categorical_encoders(X_train, encoders)
    X_test = apply_categorical_encoders(X_test, encoders)

    # 3) Resample the training set only (SMOTENC on encoded, unscaled data).
    X_train, y_train, resampled = resample_training_set(
        X_train, y_train,
        categorical_indices=categorical_indices,
        mode=resample,
        imbalance_threshold=imbalance_threshold,
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )
    balance_after = _class_counts(y_train)

    # 4) Optional scaling of continuous columns (unnecessary for LightGBM).
    scaler: Optional[StandardScaler] = None
    if scale:
        logger.info("scaling continuous features (note: not needed for tree models)")
        scaler = fit_scaler(X_train, continuous_features)
        X_train = apply_scaler(X_train, scaler, continuous_features)
        X_test = apply_scaler(X_test, scaler, continuous_features)

    # Ensure clean, aligned integer/float dtypes for LightGBM.
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    metadata = {
        "n_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_size": test_size,
        "random_state": random_state,
        "resample_mode": resample,
        "resampled": resampled,
        "scaled": bool(scale),
        "class_balance_before": balance_before,
        "class_balance_after": balance_after,
    }

    encoder_path: Optional[Path] = None
    if save_encoders:
        encoder_path = save_encoder_bundle(
            encoders, scaler, categorical_features, categorical_indices,
            models_dir, extra_metadata=metadata,
        )

    return PreprocessResult(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        feature_names=list(FEATURE_NAMES),
        categorical_features=categorical_features,
        categorical_indices=categorical_indices,
        encoders=encoders, scaler=scaler, resampled=resampled,
        class_balance_before=balance_before, class_balance_after=balance_after,
        encoder_path=encoder_path, metadata=metadata,
    )


# ===========================================================================
# CLI
# ===========================================================================
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="preprocess",
        description="Encode, split, and balance the ADIS feature matrix for training.",
    )
    p.add_argument("--input", "-i", default=str(DEFAULT_INPUT),
                   help=f"feature matrix csv/parquet (default: {DEFAULT_INPUT})")
    p.add_argument("--models-dir", "-m", default=str(DEFAULT_MODELS_DIR),
                   help=f"where to save label_encoder.pkl (default: {DEFAULT_MODELS_DIR})")
    p.add_argument("--test-size", type=float, default=0.2, help="test fraction (default: 0.2)")
    p.add_argument("--random-state", type=int, default=42, help="RNG seed (default: 42)")
    p.add_argument("--resample", choices=("auto", "smote", "none"), default="auto",
                   help="SMOTE strategy (default: auto — only if severely imbalanced)")
    p.add_argument("--imbalance-threshold", type=float, default=DEFAULT_IMBALANCE_THRESHOLD,
                   help=f"minority fraction below which auto-SMOTE triggers (default: {DEFAULT_IMBALANCE_THRESHOLD})")
    p.add_argument("--sampling-strategy", default=str(DEFAULT_SMOTE_RATIO),
                   help="SMOTE minority:majority ratio (float) or 'auto' (default: 0.5)")
    p.add_argument("--scale", action="store_true", help="scale continuous features (not needed for LightGBM)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        strategy: Any = args.sampling_strategy
        if strategy != "auto":
            strategy = float(strategy)
        result = preprocess(
            args.input,
            test_size=args.test_size,
            random_state=args.random_state,
            resample=args.resample,
            imbalance_threshold=args.imbalance_threshold,
            sampling_strategy=strategy,
            scale=args.scale,
            models_dir=args.models_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("preprocessing failed: {}", exc)
        return 1

    logger.info(
        "preprocessing complete | train={} test={} resampled={} | encoders -> {}",
        len(result.X_train), len(result.X_test), result.resampled, result.encoder_path,
    )
    logger.info(
        "train balance before={} after={}",
        result.class_balance_before, result.class_balance_after,
    )
    print(json.dumps(result.metadata, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())