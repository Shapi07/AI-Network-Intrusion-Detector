"""
predict.py
==========
Phase 7: Production prediction pipeline for the AI Network Intrusion Detector.

This module lets users — and later the Streamlit app — score *unseen*
network traffic with the model trained and evaluated in Phases 5–6.
It deliberately does not re-implement cleaning or encoding: it reuses
``src.preprocessing`` (Phase 3) and ``src.feature_engineering`` (Phase 4)
so the exact same transformations are applied at inference time as at
training time.

Responsibilities
----------------
1. Load the trained model, fitted scaler, and expected feature-name list
   from ``models/``.
2. Accept new traffic as a file path, an in-memory buffer (e.g. a
   Streamlit upload), or an already-loaded ``pd.DataFrame``.
3. Clean it with the Phase 3 pipeline and encode it with the Phase 4
   pipeline.
4. Align the engineered features to the model's expected column set and
   order — missing columns are added with default value 0, columns the
   model has never seen are dropped.
5. Run inference, attaching predictions and (when supported) class
   probabilities to the original rows.
6. Persist the result to ``predictions/`` and return it as a DataFrame.

Design notes
------------
* ``src.preprocessing.clean_dataframe`` currently requires a target/label
  column to be present (it was designed around train/test data). Real
  unseen traffic usually has no label at all. Rather than modifying the
  Phase 3 module, :func:`_clean_for_inference` transparently injects a
  placeholder label column when one is missing, runs the *unmodified*
  Phase 3 pipeline, then discards the placeholder before feature
  engineering. If a genuine label column *is* present (e.g. you're
  scoring a held-out labelled batch to spot-check the model), it is kept
  aside and attached to the output for comparison — it is never fed to
  the model as a feature.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Any, NamedTuple

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CLASS_NAMES,
    DEFAULT_PREDICTIONS_PATH,
    FEATURE_NAMES_PATH,
    MODEL_PATH,
    SCALER_PATH,
    TARGET_COLUMN,
    configure_logging,
)
from src.feature_engineering import encode_categorical
from src.preprocessing import clean_dataframe, load_csv
from src.utils import ensure_dir, timeit

logger = configure_logging(__name__)


# ──────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────

class PredictionArtifacts(NamedTuple):
    """Bundle of everything needed to run inference."""

    model: Any
    scaler: Any
    feature_names: list[str]


# ──────────────────────────────────────────────────────────────
# Artifact loading
# ──────────────────────────────────────────────────────────────

@timeit
def load_prediction_artifacts(
    *,
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    feature_names_path: Path = FEATURE_NAMES_PATH,
) -> PredictionArtifacts:
    """
    Load the trained model, fitted scaler, and expected feature names.

    Parameters
    ----------
    model_path : Path, optional
        Path to the trained model artifact. Defaults to ``config.MODEL_PATH``.
    scaler_path : Path, optional
        Path to the fitted ``StandardScaler``. Defaults to ``config.SCALER_PATH``.
    feature_names_path : Path, optional
        Path to the saved list of training feature names.
        Defaults to ``config.FEATURE_NAMES_PATH``.

    Returns
    -------
    PredictionArtifacts
        Named tuple of ``(model, scaler, feature_names)``.

    Raises
    ------
    FileNotFoundError
        If any of the three artifacts is missing on disk — this means
        Phase 5 (training) has not been run yet.
    """
    missing = [
        p for p in (model_path, scaler_path, feature_names_path) if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing model artifact(s): "
            f"{[str(p) for p in missing]}. Run training (Phase 5) first."
        )

    logger.info("📂 Loading model from %s", model_path)
    model = joblib.load(model_path)

    logger.info("📂 Loading scaler from %s", scaler_path)
    scaler = joblib.load(scaler_path)

    logger.info("📂 Loading feature names from %s", feature_names_path)
    feature_names = list(joblib.load(feature_names_path))

    logger.info(
        "✅ Artifacts loaded — model: %s | features: %d",
        type(model).__name__,
        len(feature_names),
    )
    return PredictionArtifacts(model=model, scaler=scaler, feature_names=feature_names)


# ──────────────────────────────────────────────────────────────
# Cleaning wrapper (label column is optional for inference)
# ──────────────────────────────────────────────────────────────

def _clean_for_inference(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Run the unmodified Phase 3 cleaning pipeline on data that may or may
    not carry a ground-truth label column.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw dataframe as loaded from CSV.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series | None]
        * Cleaned feature dataframe (target column removed).
        * The true labels as a ``pd.Series`` if the input had a genuine
          label column, otherwise ``None``.
    """
    try:
        df_clean, _info = clean_dataframe(df_raw.copy())
        y_true = df_clean[TARGET_COLUMN].copy()
        has_labels = True
    except KeyError:
        # No label column in the unseen data — inject a harmless
        # placeholder so the (unmodified) Phase 3 pipeline can run,
        # then discard it below.
        logger.info(
            "No target column found in input — running in label-free "
            "prediction mode."
        )
        df_tmp = df_raw.copy()
        df_tmp[TARGET_COLUMN] = 0
        df_clean, _info = clean_dataframe(df_tmp)
        y_true = None
        has_labels = False

    df_features = df_clean.drop(columns=[TARGET_COLUMN])
    return df_features, (y_true if has_labels else None)


# ──────────────────────────────────────────────────────────────
# Feature alignment
# ──────────────────────────────────────────────────────────────

def align_features(
    df_encoded: pd.DataFrame, feature_names: list[str]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Reindex an encoded dataframe to exactly match the model's expected
    feature set and order.

    * Columns the model expects but that are absent from the new data
      (e.g. a category that never appeared in this batch) are added and
      filled with ``0``.
    * Columns present in the new data but unknown to the model (e.g. a
      category never seen during training) are dropped.

    Parameters
    ----------
    df_encoded : pd.DataFrame
        Output of ``feature_engineering.encode_categorical``.
    feature_names : list[str]
        Exact column order the model was trained on.

    Returns
    -------
    tuple[pd.DataFrame, list[str], list[str]]
        * Aligned dataframe, column order guaranteed to match ``feature_names``.
        * List of columns that were added (missing → filled with 0).
        * List of columns that were dropped (unseen during training).
    """
    missing_cols = [c for c in feature_names if c not in df_encoded.columns]
    extra_cols = [c for c in df_encoded.columns if c not in feature_names]

    if missing_cols:
        logger.warning(
            "⚠️ %d expected feature(s) missing from input — filling with 0: %s",
            len(missing_cols),
            missing_cols[:10],
        )
        for col in missing_cols:
            df_encoded[col] = 0

    if extra_cols:
        logger.warning(
            "⚠️ %d unexpected column(s) in input — dropping: %s",
            len(extra_cols),
            extra_cols[:10],
        )

    aligned = df_encoded[feature_names]
    return aligned, missing_cols, extra_cols


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────

def _run_model_inference(
    X: pd.DataFrame, model: Any
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Run ``predict`` (and ``predict_proba`` when available) on already
    scaled, aligned features.

    Parameters
    ----------
    X : pd.DataFrame
        Scaled feature matrix, columns matching training order exactly.
    model : Any
        Fitted scikit-learn-compatible classifier.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        * Predicted class labels.
        * Probability of the positive ("Attack", class 1) label per row,
          or ``None`` if the model does not support ``predict_proba``.
    """
    preds = model.predict(X)

    attack_proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = list(model.classes_)
        attack_idx = classes.index(1) if 1 in classes else int(np.argmax(classes))
        attack_proba = proba[:, attack_idx]
    else:
        logger.info(
            "Model %s does not support predict_proba — probabilities omitted.",
            type(model).__name__,
        )

    return preds, attack_proba


# ──────────────────────────────────────────────────────────────
# End-to-end pipeline
# ──────────────────────────────────────────────────────────────

@timeit
def predict(
    source: str | Path | io.BytesIO | pd.DataFrame,
    *,
    artifacts: PredictionArtifacts | None = None,
    save_output: bool = True,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    End-to-end Phase 7 prediction pipeline.

    Loads (if not already loaded) → cleans (Phase 3) → encodes (Phase 4)
    → aligns → scales → predicts → saves → returns.

    Parameters
    ----------
    source : str | Path | io.BytesIO | pd.DataFrame
        Unseen network traffic: a CSV file path, an in-memory buffer
        (e.g. from a Streamlit ``file_uploader``), or an already-loaded
        raw ``pd.DataFrame``.
    artifacts : PredictionArtifacts, optional
        Pre-loaded model/scaler/feature-names bundle. If omitted, they
        are loaded from disk via :func:`load_prediction_artifacts`
        (useful for callers — e.g. a Streamlit app — that want to cache
        the artifacts across calls instead of reloading them each time).
    save_output : bool, optional
        Whether to persist the results CSV to disk. Defaults to ``True``.
    output_path : str | Path, optional
        Destination for the results CSV. Defaults to
        ``config.DEFAULT_PREDICTIONS_PATH``.

    Returns
    -------
    pd.DataFrame
        One row per input record, in the original row order, with columns:

        * ``prediction`` — raw class label (``0`` = Normal, ``1`` = Attack).
        * ``prediction_label`` — human-readable class name.
        * ``attack_probability`` — model confidence the row is an attack
          (only present if the model supports ``predict_proba``).
        * ``true_label`` / ``true_label_name`` — only present if the
          input already contained a genuine ground-truth label column.

        Metadata about the run (row/feature counts, columns that were
        added or dropped during alignment, whether ground truth was
        available, and the output file path) is attached to
        ``result.attrs`` for callers that want it without changing the
        DataFrame's shape.

    Raises
    ------
    FileNotFoundError
        If ``artifacts`` is not provided and the trained model artifacts
        are missing from ``models/``.
    ValueError
        If the input CSV is empty or cannot be parsed.
    """
    if artifacts is None:
        artifacts = load_prediction_artifacts()
    model, scaler, feature_names = artifacts

    # 1. Load
    if isinstance(source, pd.DataFrame):
        df_raw = source.copy()
        logger.info("📂 Using in-memory DataFrame: %s rows", len(df_raw))
    else:
        df_raw = load_csv(source)

    n_input_rows = len(df_raw)

    # 2. Clean (Phase 3, reused as-is)
    df_features, y_true = _clean_for_inference(df_raw)
    has_labels = y_true is not None

    # 3. Encode categorical features (Phase 4, reused as-is)
    df_encoded = encode_categorical(df_features)

    # 4. Align to the model's expected feature set/order
    X_aligned, missing_cols, extra_cols = align_features(df_encoded, feature_names)

    # 5. Scale with the artifact fitted during training
    X_scaled = pd.DataFrame(
        scaler.transform(X_aligned),
        columns=X_aligned.columns,
        index=X_aligned.index,
    )

    # 6. Predict
    logger.info("🔮 Running inference on %d row(s)...", len(X_scaled))
    preds, attack_proba = _run_model_inference(X_scaled, model)

    # 7. Assemble results
    results = pd.DataFrame(index=df_features.index)
    results["prediction"] = preds
    results["prediction_label"] = [
        CLASS_NAMES[1] if p == 1 else CLASS_NAMES[0] for p in preds
    ]
    if attack_proba is not None:
        results["attack_probability"] = np.round(attack_proba, 4)
    if has_labels:
        results["true_label"] = y_true.values
        results["true_label_name"] = [
            CLASS_NAMES[1] if v == 1 else CLASS_NAMES[0] for v in y_true.values
        ]

    n_attack = int((results["prediction"] == 1).sum())
    n_normal = len(results) - n_attack
    logger.info(
        "🎯 Prediction complete — Normal: %d | Attack: %d", n_normal, n_attack
    )

    # 8. Persist
    resolved_output_path = Path(output_path) if output_path else DEFAULT_PREDICTIONS_PATH
    if save_output:
        ensure_dir(resolved_output_path.parent)
        results.to_csv(resolved_output_path, index=True)
        logger.info("💾 Saved predictions → %s", resolved_output_path)

    # 9. Attach run metadata without altering the DataFrame's columns/shape
    results.attrs.update(
        {
            "n_input_rows": n_input_rows,
            "n_normal": n_normal,
            "n_attack": n_attack,
            "has_ground_truth": has_labels,
            "missing_features": missing_cols,
            "extra_features": extra_cols,
            "output_path": str(resolved_output_path) if save_output else None,
        }
    )

    return results


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AINID Phase 7 — run the trained model on unseen network traffic."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the CSV file with new traffic."
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Where to write the predictions CSV "
        f"(default: {DEFAULT_PREDICTIONS_PATH}).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't write results to disk, just print a summary.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    results = predict(
        args.input,
        save_output=not args.no_save,
        output_path=args.output,
    )

    print("\n" + "=" * 50)
    print("✅ PHASE 7 — PREDICTION COMPLETE")
    print(f"📄 Input rows       : {results.attrs['n_input_rows']}")
    print(f"🟢 Normal            : {results.attrs['n_normal']}")
    print(f"🔴 Attack            : {results.attrs['n_attack']}")
    print(f"🏷️  Ground truth found: {results.attrs['has_ground_truth']}")
    if results.attrs["missing_features"]:
        print(f"⚠️  Missing features  : {len(results.attrs['missing_features'])}")
    if results.attrs["output_path"]:
        print(f"💾 Saved to          : {results.attrs['output_path']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
