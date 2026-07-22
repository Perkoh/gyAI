"""
ml/training/feature_builder.py
==============================
gyAI — AI-Powered Domain Intelligence System (ADIS)

Build the model training feature matrix from the raw labelled CSV
(``ml/data/raw/domains.csv``) by running the **exact same** feature
extraction pipeline that serves live requests.

Why this matters (blueprint FLAG 7)
-----------------------------------
Features are never pre-computed by hand. This builder calls
``features.assemble_feature_dict`` — the identical code path used at
inference time by the ``/analyze`` route — so the representation the model
trains on is guaranteed to match the representation it sees in production.
Any drift there silently destroys accuracy, so there is a single source of
truth: the ``features`` package.

What it produces
----------------
A tabular dataset with, in order:

    domain                | the raw domain string (kept for traceability/debug)
    <48 feature columns>  | in canonical ``features.FEATURE_NAMES`` order
    label                 | 0 = safe/benign, 1 = malicious/phishing

Categorical features (``tld``, ``whois_country``) are left as **raw strings**
here on purpose: label-encoding, scaling and class-imbalance handling are the
job of ``ml/training/preprocess.py`` (blueprint build order step 9), which
fits its encoders on this matrix and saves them alongside the model.

Network features and training
-----------------------------
Network features (31–48) require live DNS/WHOIS lookups. Extracting them for a
large corpus is slow and rate-limited (blueprint FLAG 4), and many historical
phishing domains are already dead — so their network fields fall back to
defaults. Training on those defaults risks *availability leakage* (the model
learns "no network data == phishing", which will not hold for a live phishing
site). Consider this when choosing ``--no-network``:

    * ``--no-network``  : structural features only, network columns defaulted.
                          Fast, reproducible, avoids leakage. Good default for
                          iteration and for corpora of dead URLs.
    * network enabled   : full 48-feature parity with production. Use a live,
                          freshly-collected corpus and expect a long run; the
                          on-disk cache (``--cache``) makes re-runs cheap and
                          spares WHOIS servers.

Usage
-----
    python -m ml.training.feature_builder \
        --input ml/data/raw/domains.csv \
        --output ml/data/processed/ \
        --no-network --format csv

Programmatic:
    from ml.training.feature_builder import build_and_save, load_raw_dataset
"""

from __future__ import annotations

import argparse
import json
import os
import shelve
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

# The one and only feature pipeline — shared with inference.
from features import (
    CATEGORICAL_FEATURES,
    DEFAULT_NETWORK_TIMEOUT,
    FEATURE_COUNT,
    FEATURE_NAMES,
    FeatureExtractionError,
    assemble_feature_dict,
)

# Structured logging (loguru per blueprint §8.1) with a stdlib fallback.
try:  # pragma: no cover - trivial import shim
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    class _BraceLogger:
        def __init__(self, name: str) -> None:
            self._log = logging.getLogger(name)

        def _emit(self, level, msg, args, kwargs):
            if self._log.isEnabledFor(level):
                try:
                    text = msg.format(*args, **kwargs) if (args or kwargs) else msg
                except Exception:
                    text = msg
                self._log.log(level, text)

        def debug(self, m, *a, **k): self._emit(logging.DEBUG, m, a, k)
        def info(self, m, *a, **k): self._emit(logging.INFO, m, a, k)
        def warning(self, m, *a, **k): self._emit(logging.WARNING, m, a, k)
        def error(self, m, *a, **k): self._emit(logging.ERROR, m, a, k)

    logger = _BraceLogger("adis.feature_builder")


BUILDER_VERSION = "1.0.0"

DEFAULT_INPUT = Path("ml/data/raw/domains.csv")
DEFAULT_OUTPUT_DIR = Path("ml/data/processed")
DEFAULT_BASENAME = "features"

DOMAIN_COLUMN = "domain"
LABEL_COLUMN = "label"
VALID_LABELS = frozenset({0, 1})


# ===========================================================================
# 1. Raw dataset loading & validation
# ===========================================================================
def load_raw_dataset(csv_path: str | os.PathLike) -> pd.DataFrame:
    """Load and validate the raw ``domain,label`` CSV.

    Cleans the data into a canonical form:
      * requires ``domain`` and ``label`` columns (case-insensitive header),
      * lowercases/strips domains and drops empties,
      * coerces labels to integers and keeps only 0/1,
      * de-duplicates domains (keeping the first), warning on any domain that
        appears with conflicting labels.

    Returns a DataFrame with exactly two columns: ``domain``, ``label``.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"raw dataset not found at {path}. Place your CSV there "
            f"(blueprint §7.4) or pass --input."
        )

    df = pd.read_csv(path, dtype={LABEL_COLUMN: "object"})
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = {DOMAIN_COLUMN, LABEL_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {sorted(missing)}. "
            f"Expected header: '{DOMAIN_COLUMN},{LABEL_COLUMN}' (blueprint §7.1)."
        )

    df = df[[DOMAIN_COLUMN, LABEL_COLUMN]].copy()
    n_start = len(df)

    # --- Clean domains ------------------------------------------------------
    df[DOMAIN_COLUMN] = df[DOMAIN_COLUMN].astype("string").str.strip().str.lower()
    df = df[df[DOMAIN_COLUMN].notna() & (df[DOMAIN_COLUMN] != "")]

    # --- Clean labels -------------------------------------------------------
    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
    bad_label = df[LABEL_COLUMN].isna()
    if bad_label.any():
        logger.warning("dropping {} row(s) with non-numeric labels", int(bad_label.sum()))
    df = df[~bad_label]
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    out_of_range = ~df[LABEL_COLUMN].isin(VALID_LABELS)
    if out_of_range.any():
        logger.warning(
            "dropping {} row(s) with labels outside {{0,1}}", int(out_of_range.sum())
        )
    df = df[~out_of_range]

    # --- De-duplicate, detecting label conflicts ---------------------------
    conflict_mask = df.groupby(DOMAIN_COLUMN)[LABEL_COLUMN].transform("nunique") > 1
    n_conflicts = df.loc[conflict_mask, DOMAIN_COLUMN].nunique()
    if n_conflicts:
        logger.warning(
            "{} domain(s) appear with conflicting labels; keeping first occurrence",
            n_conflicts,
        )
    df = df.drop_duplicates(subset=DOMAIN_COLUMN, keep="first").reset_index(drop=True)

    logger.info(
        "loaded {} usable rows from {} (from {} raw rows)", len(df), path, n_start
    )
    if df.empty:
        raise ValueError(f"no valid rows remained after cleaning {path}")

    _log_class_balance(df)
    return df


def _log_class_balance(df: pd.DataFrame) -> None:
    counts = df[LABEL_COLUMN].value_counts().to_dict()
    benign = int(counts.get(0, 0))
    malicious = int(counts.get(1, 0))
    total = benign + malicious
    if total:
        logger.info(
            "class balance: benign={} ({:.1%}) | malicious={} ({:.1%})",
            benign, benign / total, malicious, malicious / total,
        )


# ===========================================================================
# 2. Per-domain extraction
# ===========================================================================
def _extract_one(
    domain: str, include_network: bool, network_timeout: float
) -> Optional["dict[str, Any]"]:
    """Extract one domain's 48 features. Returns None on extraction failure."""
    try:
        return dict(
            assemble_feature_dict(
                domain,
                include_network=include_network,
                network_timeout=network_timeout,
            )
        )
    except FeatureExtractionError as exc:
        logger.debug("extraction failed for {!r}: {}", domain, exc)
        return None
    except Exception as exc:  # never let one bad row kill the whole build
        logger.warning("unexpected error extracting {!r}: {}", domain, exc)
        return None


def _cache_key(domain: str, include_network: bool) -> str:
    return f"{int(include_network)}:{domain}"


# ===========================================================================
# 3. Feature matrix construction
# ===========================================================================
def build_feature_matrix(
    df: pd.DataFrame,
    *,
    include_network: bool = True,
    max_workers: Optional[int] = None,
    network_timeout: float = DEFAULT_NETWORK_TIMEOUT,
    on_error: str = "skip",
    cache_path: Optional[str | os.PathLike] = None,
    progress_every: int = 500,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Run feature extraction across the dataset and return the feature matrix.

    Args:
        df: Cleaned dataset from :func:`load_raw_dataset` (columns domain,label).
        include_network: Extract network features via live DNS/WHOIS. When
            False, network columns take their documented defaults (fast, offline).
        max_workers: Thread-pool size. Network extraction is I/O-bound, so a
            pool helps; structural-only runs are CPU-cheap and run serially.
            Defaults to 16 when network is enabled, else 1.
        network_timeout: Per-domain network time budget (seconds).
        on_error: 'skip' (default) drops rows that fail to extract; 'raise'
            aborts on the first failure.
        cache_path: Optional on-disk (shelve) cache of extracted features,
            keyed by domain. Makes re-runs cheap and spares WHOIS servers.
        progress_every: Log a progress line every N unique domains.
        limit: Process at most this many rows (handy for smoke tests).

    Returns:
        DataFrame: ['domain', *FEATURE_NAMES, 'label'], categoricals as strings.
    """
    if on_error not in {"skip", "raise"}:
        raise ValueError(f"on_error must be 'skip' or 'raise'; got {on_error!r}")

    work = df if limit is None else df.head(limit)
    unique_domains = list(dict.fromkeys(work[DOMAIN_COLUMN].tolist()))
    if max_workers is None:
        max_workers = 16 if include_network else 1

    logger.info(
        "extracting features for {} unique domain(s) "
        "(network={}, workers={}, cache={})",
        len(unique_domains), include_network, max_workers, bool(cache_path),
    )

    results: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    start = time.perf_counter()

    cache = shelve.open(str(cache_path)) if cache_path else None
    try:
        # Serve what we can from cache first.
        pending: list[str] = []
        for d in unique_domains:
            if cache is not None:
                cached = cache.get(_cache_key(d, include_network))
                if cached is not None:
                    results[d] = cached
                    continue
            pending.append(d)

        if cache is not None:
            logger.info("cache hits: {} | to compute: {}", len(results), len(pending))

        done = 0

        def _record(domain: str, feats: Optional[dict]) -> None:
            nonlocal done
            done += 1
            if feats is None:
                failures.append(domain)
                if on_error == "raise":
                    raise FeatureExtractionError(
                        f"feature extraction failed for {domain!r} (on_error='raise')"
                    )
            else:
                results[domain] = feats
                if cache is not None:  # writes happen only in the main thread
                    cache[_cache_key(domain, include_network)] = feats
            if done % progress_every == 0 or done == len(pending):
                _log_progress(done, len(pending), start)

        # Structural-only or single-worker: simple serial loop.
        if max_workers <= 1 or not include_network:
            for d in pending:
                _record(d, _extract_one(d, include_network, network_timeout))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_domain = {
                    pool.submit(_extract_one, d, include_network, network_timeout): d
                    for d in pending
                }
                for future in as_completed(future_to_domain):
                    d = future_to_domain[future]
                    try:
                        feats = future.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning("worker crashed on {!r}: {}", d, exc)
                        feats = None
                    _record(d, feats)
    finally:
        if cache is not None:
            cache.close()

    elapsed = time.perf_counter() - start
    logger.info(
        "extraction complete: {} succeeded, {} failed in {:.1f}s",
        len(results), len(failures), elapsed,
    )

    matrix = _assemble_dataframe(work, results)
    if matrix.empty:
        raise ValueError(
            "no rows survived feature extraction; check the dataset and logs"
        )
    return matrix


def _log_progress(done: int, total: int, start: float) -> None:
    elapsed = time.perf_counter() - start
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else float("nan")
    logger.info(
        "  progress {}/{} ({:.1%}) | {:.1f}/s | ETA {:.0f}s",
        done, total, (done / total if total else 1.0), rate, remaining,
    )


def _assemble_dataframe(
    work: pd.DataFrame, results: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    """Join extracted features back onto their labels, preserving row order."""
    kept = work[work[DOMAIN_COLUMN].isin(results)]
    if kept.empty:
        return pd.DataFrame(columns=[DOMAIN_COLUMN, *FEATURE_NAMES, LABEL_COLUMN])

    feature_rows = [results[d] for d in kept[DOMAIN_COLUMN]]
    matrix = pd.DataFrame(feature_rows, columns=list(FEATURE_NAMES))
    matrix.insert(0, DOMAIN_COLUMN, kept[DOMAIN_COLUMN].to_numpy())
    matrix[LABEL_COLUMN] = kept[LABEL_COLUMN].to_numpy()

    # Normalise dtypes: categoricals stay strings (encoded later in preprocess);
    # every other feature becomes numeric (booleans -> 0/1).
    for col in FEATURE_NAMES:
        if col in CATEGORICAL_FEATURES:
            matrix[col] = matrix[col].astype("string").fillna("unknown")
        else:
            matrix[col] = pd.to_numeric(matrix[col], errors="coerce").fillna(0)
    matrix[LABEL_COLUMN] = matrix[LABEL_COLUMN].astype(int)

    if len(matrix.columns) != FEATURE_COUNT + 2:  # domain + 48 + label
        raise ValueError(
            f"assembled matrix has {len(matrix.columns)} columns, "
            f"expected {FEATURE_COUNT + 2}"
        )
    return matrix.reset_index(drop=True)


# ===========================================================================
# 4. Persistence
# ===========================================================================
def save_feature_matrix(
    matrix: pd.DataFrame,
    output_dir: str | os.PathLike = DEFAULT_OUTPUT_DIR,
    *,
    basename: str = DEFAULT_BASENAME,
    fmt: str = "parquet",
    include_network: bool = True,
    source_csv: Optional[str] = None,
) -> Path:
    """Write the matrix (+ a JSON metadata sidecar) to ``output_dir``.

    Prefers Parquet (compact, typed) and transparently falls back to CSV when
    no Parquet engine (pyarrow/fastparquet) is installed. Returns the data path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = out_dir / f"{basename}.{fmt}"
    if fmt == "parquet":
        try:
            matrix.to_parquet(data_path, index=False)
        except Exception as exc:  # engine missing or write error -> CSV fallback
            data_path = out_dir / f"{basename}.csv"
            logger.warning("parquet unavailable ({}); writing CSV instead", exc)
            matrix.to_csv(data_path, index=False)
    elif fmt == "csv":
        matrix.to_csv(data_path, index=False)
    else:
        raise ValueError(f"unsupported format {fmt!r}; use 'parquet' or 'csv'")

    meta = _build_metadata(matrix, include_network=include_network, source_csv=source_csv)
    meta_path = out_dir / f"{basename}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info("wrote {} rows -> {}", len(matrix), data_path)
    logger.info("wrote metadata -> {}", meta_path)
    return data_path


def _build_metadata(
    matrix: pd.DataFrame, *, include_network: bool, source_csv: Optional[str]
) -> dict[str, Any]:
    labels = matrix[LABEL_COLUMN]
    net_col = "network_features_available"
    net_rate = (
        float(pd.to_numeric(matrix[net_col], errors="coerce").fillna(0).mean())
        if net_col in matrix.columns
        else 0.0
    )
    return {
        "builder_version": BUILDER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(source_csv) if source_csv else None,
        "n_rows": int(len(matrix)),
        "n_features": FEATURE_COUNT,
        "feature_names": list(FEATURE_NAMES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "include_network": bool(include_network),
        "network_features_available_rate": round(net_rate, 4),
        "class_balance": {
            "benign": int((labels == 0).sum()),
            "malicious": int((labels == 1).sum()),
        },
    }


# ===========================================================================
# 5. Orchestration + CLI
# ===========================================================================
def build_and_save(
    input_csv: str | os.PathLike = DEFAULT_INPUT,
    output_dir: str | os.PathLike = DEFAULT_OUTPUT_DIR,
    *,
    include_network: bool = True,
    max_workers: Optional[int] = None,
    network_timeout: float = DEFAULT_NETWORK_TIMEOUT,
    on_error: str = "skip",
    cache_path: Optional[str | os.PathLike] = None,
    fmt: str = "parquet",
    limit: Optional[int] = None,
    basename: str = DEFAULT_BASENAME,
) -> Path:
    """End-to-end: load raw CSV -> build feature matrix -> persist to disk."""
    df = load_raw_dataset(input_csv)
    matrix = build_feature_matrix(
        df,
        include_network=include_network,
        max_workers=max_workers,
        network_timeout=network_timeout,
        on_error=on_error,
        cache_path=cache_path,
        limit=limit,
    )
    return save_feature_matrix(
        matrix,
        output_dir,
        basename=basename,
        fmt=fmt,
        include_network=include_network,
        source_csv=str(input_csv),
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="feature_builder",
        description="Build the ADIS training feature matrix from the raw domain CSV.",
    )
    parser.add_argument("--input", "-i", default=str(DEFAULT_INPUT),
                        help=f"raw CSV path (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT_DIR),
                        help=f"output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--format", "-f", choices=("parquet", "csv"), default="parquet",
                        help="output format (default: parquet, falls back to csv)")
    parser.add_argument("--no-network", action="store_true",
                        help="structural features only; network columns defaulted")
    parser.add_argument("--workers", type=int, default=None,
                        help="thread-pool size (default: 16 with network, else 1)")
    parser.add_argument("--network-timeout", type=float, default=DEFAULT_NETWORK_TIMEOUT,
                        help=f"per-domain network budget in seconds (default: {DEFAULT_NETWORK_TIMEOUT})")
    parser.add_argument("--on-error", choices=("skip", "raise"), default="skip",
                        help="behaviour when a domain fails to extract (default: skip)")
    parser.add_argument("--cache", default=None,
                        help="optional on-disk feature cache path (shelve)")
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most N rows (smoke testing)")
    parser.add_argument("--basename", default=DEFAULT_BASENAME,
                        help=f"output file basename (default: {DEFAULT_BASENAME})")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        out_path = build_and_save(
            input_csv=args.input,
            output_dir=args.output,
            include_network=not args.no_network,
            max_workers=args.workers,
            network_timeout=args.network_timeout,
            on_error=args.on_error,
            cache_path=args.cache,
            fmt=args.format,
            limit=args.limit,
            basename=args.basename,
        )
    except (FileNotFoundError, ValueError, FeatureExtractionError) as exc:
        logger.error("feature build failed: {}", exc)
        return 1
    logger.info("done: {}", out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
