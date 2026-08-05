"""
preprocessing.py
================
Dataset loading, validation, and cleaning for the AI Network Intrusion Detector.

This module handles the full ingestion pipeline:

1. **Load** — Read a CSV file (local path or uploaded bytes) into a DataFrame.
2. **Detect** — Auto-identify whether the file is UNSW-NB15 or CIC-IDS2017.
3. **Validate** — Check for required columns, unexpected dtypes, and extreme missingness.
4. **Clean** — Strip whitespace from column names and string values, drop
               identifier-only columns, coerce numeric columns, and handle
               infinite / NaN values.
5. **Report** — Return a structured summary dict for downstream logging and the UI.

Design decisions
----------------
* All public functions accept a ``pd.DataFrame`` *or* a file path so they can
  be composed freely (CLI, Streamlit upload, pytest fixtures).
* Chunked reading (``chunksize``) is used for large files to avoid OOM errors
  on machines with limited RAM.  The chunks are concatenated after basic
  per-chunk validation.
* Cleaning is non-destructive by default — original columns are never silently
  renamed without logging.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.config import (
    COLUMNS_TO_DROP,
    PROCESSED_DATASET_PATH,
    TARGET_COLUMN,
    configure_logging,
)
from src.utils import dataframe_memory_report, ensure_dir, timeit

logger = configure_logging(__name__)

# ──────────────────────────────────────────────────────────────
# Dataset format signatures
# ──────────────────────────────────────────────────────────────

# A handful of columns that uniquely identify each dataset variant.
# We only need to recognise a few — full column lists are in the datasets.
_UNSW_NB15_SIGNATURE_COLS: frozenset[str] = frozenset(
    {"proto", "service", "state", "attack_cat", "label"}
)

_CIC_IDS2017_SIGNATURE_COLS: frozenset[str] = frozenset(
    {
        "destination port",
        "flow duration",
        "total fwd packets",
        "label",
    }
)

# Chunk size for reading large files (number of rows per chunk)
_CHUNK_SIZE: int = 100_000


# ──────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────

class DatasetInfo(NamedTuple):
    """Structured summary returned by :func:`load_dataset`."""

    format: str                    # "UNSW-NB15" | "CIC-IDS2017" | "unknown"
    n_rows: int
    n_cols: int
    missing_cells: int
    missing_pct: float             # 0.0 – 100.0
    duplicate_rows: int
    target_distribution: dict[str, int]  # {class_label: count}
    memory_mb: float


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading/trailing whitespace from column names and
    convert them to lower-case for uniform downstream access.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with potentially dirty column names.

    Returns
    -------
    pd.DataFrame
        Same dataframe with clean column names (mutated in place).
    """
    original = list(df.columns)
    df.columns = [c.strip().lower() for c in df.columns]
    renamed = [
        (o, n) for o, n in zip(original, df.columns) if o != n
    ]
    if renamed:
        logger.debug(
            "Normalised %d column name(s): %s",
            len(renamed),
            renamed[:5],
        )
    return df


def _detect_format(df: pd.DataFrame) -> str:
    """
    Identify the dataset format by matching signature column sets.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with already-normalised (lower-cased) column names.

    Returns
    -------
    str
        One of ``"UNSW-NB15"``, ``"CIC-IDS2017"``, or ``"unknown"``.
    """
    cols = frozenset(df.columns)

    if _UNSW_NB15_SIGNATURE_COLS.issubset(cols):
        return "UNSW-NB15"
    if _CIC_IDS2017_SIGNATURE_COLS.issubset(cols):
        return "CIC-IDS2017"
    return "unknown"


def _validate_target_column(df: pd.DataFrame, fmt: str) -> pd.DataFrame:
    """
    Ensure the target column exists, renaming dataset-specific
    variants to the canonical ``TARGET_COLUMN`` name if necessary.

    CIC-IDS2017 uses a ``"label"`` column whose values are class
    names (e.g. ``"BENIGN"``, ``"DDoS"``).  UNSW-NB15 uses a
    binary integer ``"label"`` column (0 / 1).  Both are normalised
    here so downstream code always operates on the same column name.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    fmt : str
        Detected dataset format string.

    Returns
    -------
    pd.DataFrame
        DataFrame with a guaranteed ``TARGET_COLUMN`` column.

    Raises
    ------
    KeyError
        If no recognisable target column can be found.
    """
    if TARGET_COLUMN in df.columns:
        return df

    # Fallback search — accept any column containing "label"
    candidates = [c for c in df.columns if "label" in c]
    if candidates:
        chosen = candidates[0]
        df = df.rename(columns={chosen: TARGET_COLUMN})
        logger.warning(
            "Target column '%s' not found; using '%s' instead.",
            TARGET_COLUMN,
            chosen,
        )
        return df

    raise KeyError(
        f"No target column found. Expected '{TARGET_COLUMN}'. "
        f"Available columns: {list(df.columns)}"
    )


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to cast object-dtype columns to numeric where possible.

    Mixed-type columns (e.g. a numeric column with a stray string
    header repeated mid-file) are coerced with ``errors='coerce'``,
    turning non-parseable values into NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        DataFrame with numeric columns properly typed.
    """
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    # Never coerce the target column — it may be a string label
    obj_cols = [c for c in obj_cols if c != TARGET_COLUMN]

    converted: list[str] = []
    for col in obj_cols:
        coerced = pd.to_numeric(df[col], errors="coerce")
        # Only apply conversion if at least 90 % of values parsed successfully
        non_null_pct = coerced.notna().mean()
        if non_null_pct >= 0.90:
            df[col] = coerced
            converted.append(col)

    if converted:
        logger.debug("Coerced %d object column(s) to numeric.", len(converted))

    return df


def _handle_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace ``±Inf`` values with NaN so they are handled uniformly.

    Infinite values appear in network datasets when flow duration
    is zero and per-packet metrics divide by it.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        DataFrame with infinite values replaced by NaN.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_mask = df[numeric_cols].isin([np.inf, -np.inf])
    n_inf = int(inf_mask.values.sum())

    if n_inf > 0:
        df[numeric_cols] = df[numeric_cols].replace(
            [np.inf, -np.inf], np.nan
        )
        logger.warning(
            "Replaced %d infinite value(s) with NaN.", n_inf
        )

    return df


def _build_dataset_info(df: pd.DataFrame, fmt: str) -> DatasetInfo:
    """
    Compute summary statistics on the loaded dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    fmt : str
        Detected format string.

    Returns
    -------
    DatasetInfo
        Named tuple of dataset statistics.
    """
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    missing_pct = round(100.0 * missing_cells / total_cells, 4) if total_cells else 0.0
    duplicate_rows = int(df.duplicated().sum())
    mem_mb = round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2)

    # Target distribution (works for both binary int and string labels)
    if TARGET_COLUMN in df.columns:
        target_dist = df[TARGET_COLUMN].value_counts().to_dict()
        # Ensure keys are plain Python types (not numpy) for JSON serialisation
        target_dist = {str(k): int(v) for k, v in target_dist.items()}
    else:
        target_dist = {}

    return DatasetInfo(
        format=fmt,
        n_rows=df.shape[0],
        n_cols=df.shape[1],
        missing_cells=missing_cells,
        missing_pct=missing_pct,
        duplicate_rows=duplicate_rows,
        target_distribution=target_dist,
        memory_mb=mem_mb,
    )


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

@timeit
def load_csv(
    source: str | Path | io.BytesIO,
    *,
    chunksize: int = _CHUNK_SIZE,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Load a network intrusion CSV file into a Pandas DataFrame.

    Supports:
    * A file-system path (``str`` or ``pathlib.Path``).
    * An in-memory ``io.BytesIO`` object (e.g. from a Streamlit file uploader).

    For large files the CSV is read in chunks of *chunksize* rows and
    concatenated.  This prevents out-of-memory errors on machines with
    limited RAM.

    Parameters
    ----------
    source : str | Path | io.BytesIO
        CSV data source.
    chunksize : int, optional
        Number of rows per chunk when reading large files.
        Defaults to 100 000.
    encoding : str, optional
        File encoding.  Defaults to ``"utf-8"``.

    Returns
    -------
    pd.DataFrame
        Raw (un-cleaned) dataframe.

    Raises
    ------
    FileNotFoundError
        If *source* is a path that does not exist.
    ValueError
        If the file is empty or cannot be parsed as CSV.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        logger.info("📂 Loading dataset from: %s", path)
        source_label = str(path.name)
    else:
        logger.info("📂 Loading dataset from in-memory buffer.")
        source_label = "<uploaded file>"

    try:
        chunks: list[pd.DataFrame] = []
        reader = pd.read_csv(
            source,
            chunksize=chunksize,
            encoding=encoding,
            low_memory=False,
        )
        for i, chunk in enumerate(reader):
            chunks.append(chunk)
            if (i + 1) % 5 == 0:
                logger.debug(
                    "  … read %d rows so far", (i + 1) * chunksize
                )

        if not chunks:
            raise ValueError(f"CSV file is empty: {source_label}")

        df = pd.concat(chunks, ignore_index=True)
        logger.info(
            "✅ Loaded %s  →  %s",
            source_label,
            dataframe_memory_report(df),
        )
        return df

    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty or malformed: {source_label}") from exc
    except UnicodeDecodeError:
        # Retry with latin-1 which can decode any byte sequence
        logger.warning(
            "UTF-8 decoding failed for '%s', retrying with latin-1.",
            source_label,
        )
        if isinstance(source, io.BytesIO):
            source.seek(0)
        return load_csv(source, chunksize=chunksize, encoding="latin-1")


def detect_dataset_format(df: pd.DataFrame) -> str:
    """
    Return the detected dataset format after normalising column names.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or cleaned dataframe.

    Returns
    -------
    str
        One of ``"UNSW-NB15"``, ``"CIC-IDS2017"``, or ``"unknown"``.
    """
    df_norm = _normalise_columns(df.copy())
    fmt = _detect_format(df_norm)
    logger.info("🔍 Detected dataset format: %s", fmt)
    return fmt


@timeit
def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, DatasetInfo]:
    """
    Apply the full cleaning pipeline to a raw dataframe.

    Steps applied (in order):

    1. Normalise column names (strip, lower-case).
    2. Detect dataset format.
    3. Validate and standardise the target column.
    4. Drop identifier-only columns (``id``, ``attack_cat`` etc.).
    5. Coerce string-encoded numeric columns to float.
    6. Replace ±Inf with NaN.
    7. Drop fully-duplicate rows.
    8. Log and return a ``DatasetInfo`` summary.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from :func:`load_csv`.

    Returns
    -------
    tuple[pd.DataFrame, DatasetInfo]
        * Cleaned dataframe.
        * Dataset information summary.

    Raises
    ------
    KeyError
        If no target column can be found.
    """
    logger.info("🧹 Starting cleaning pipeline …")

    # 1. Normalise column names
    df = _normalise_columns(df)

    # 2. Detect format
    fmt = _detect_format(df)
    logger.info("   Format detected: %s", fmt)

    # 3. Validate target column
    df = _validate_target_column(df, fmt)

    # 4. Drop identifier / non-feature columns
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info("   Dropped %d identifier column(s): %s", len(cols_to_drop), cols_to_drop)

    # 5. Coerce object columns that are actually numeric
    df = _coerce_numeric_columns(df)

    # 6. Replace infinities
    df = _handle_infinite_values(df)

    # 7. Drop duplicate rows
    n_before = len(df)
    df = df.drop_duplicates()
    n_dupes = n_before - len(df)
    if n_dupes:
        logger.info("   Removed %d duplicate row(s).", n_dupes)

    # Build summary
    info = _build_dataset_info(df, fmt)

    logger.info(
        "✅ Cleaning complete — %d rows × %d cols | "
        "missing: %.2f%% | dupes removed: %d",
        info.n_rows,
        info.n_cols,
        info.missing_pct,
        n_dupes,
    )

    return df, info


@timeit
def load_dataset(
    source: str | Path | io.BytesIO,
    *,
    save_processed: bool = False,
) -> tuple[pd.DataFrame, DatasetInfo]:
    """
    End-to-end dataset loading convenience function.

    Combines :func:`load_csv` and :func:`clean_dataframe` into a
    single call.  Optionally saves the cleaned dataset to
    ``data/processed/`` for faster subsequent loads.

    Parameters
    ----------
    source : str | Path | io.BytesIO
        CSV data source — path or in-memory buffer.
    save_processed : bool, optional
        If ``True``, persist the cleaned dataframe as a CSV in
        ``data/processed/``.  Defaults to ``False``.

    Returns
    -------
    tuple[pd.DataFrame, DatasetInfo]
        * Fully cleaned dataframe ready for feature engineering.
        * Dataset information summary.
    """
    raw_df = load_csv(source)
    clean_df, info = clean_dataframe(raw_df)

    if save_processed:
        ensure_dir(PROCESSED_DATASET_PATH.parent)
        clean_df.to_csv(PROCESSED_DATASET_PATH, index=False)
        logger.info("💾 Saved processed dataset → %s", PROCESSED_DATASET_PATH)

    return clean_df, info


def get_missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a per-column missing-value report.

    Parameters
    ----------
    df : pd.DataFrame
        Any dataframe (raw or cleaned).

    Returns
    -------
    pd.DataFrame
        A dataframe with columns ``["column", "missing_count", "missing_pct"]``
        sorted descending by ``missing_pct``, containing only columns
        with at least one missing value.
    """
    missing_count = df.isna().sum()
    missing_pct = 100.0 * missing_count / len(df)

    report = pd.DataFrame(
        {
            "column": missing_count.index,
            "missing_count": missing_count.values,
            "missing_pct": missing_pct.values,
        }
    )
    report = report[report["missing_count"] > 0].sort_values(
        "missing_pct", ascending=False
    ).reset_index(drop=True)

    return report
