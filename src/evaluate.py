"""
evaluate.py
===========
Phase 6: Detailed model evaluation and quality metrics.

Responsibilities
----------------
* Loading the trained model and test data.
* Calculating classification metrics (Accuracy, Precision, Recall, F1-score).
* Building the confusion matrix.
* Saving the detailed evaluation report to a file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import MODEL_PATH, REPORTS_DIR, configure_logging
from src.utils import ensure_dir, save_json

logger = configure_logging(__name__)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_report: bool = True,
) -> dict[str, Any]:
    """
    Calculates detailed quality metrics for the model on the test set.

    Parameters
    ----------
    model : Any
        Trained model (e.g., RandomForestClassifier).
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True class labels for the test set.
    save_report : bool
        Whether to save the report to a JSON file.

    Returns
    -------
    dict[str, Any]
        Dictionary with metrics (accuracy, precision, recall, f1, confusion_matrix).
    """
    logger.info("📊 Running detailed model evaluation (Phase 6)...")

    # 1. Generate model predictions
    y_pred = model.predict(X_test)

    # 2. Compute basic metrics
    acc = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    # 3. Compute Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # 4. Compile full evaluation report
    evaluation_results = {
        "accuracy": round(acc, 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
    }

    logger.info(
        "🎯 Evaluation results -> Accuracy: %.4f | F1-score: %.4f", acc, f1
    )
    logger.info(
        "📌 Confusion Matrix: TN=%d, FP=%d, FN=%d, TP=%d", tn, fp, fn, tp
    )

    # 5. Save report to disk if requested
    if save_report:
        reports_path = ensure_dir(REPORTS_DIR)
        report_file = reports_path / "evaluation_metrics.json"
        save_json(evaluation_results, report_file)

    return evaluation_results


def load_and_evaluate(X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """
    Loads the saved model from disk and triggers evaluation.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"❌ Model not found at: {MODEL_PATH}. Run training first (Phase 5)."
        )

    logger.info("📂 Loading model from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    return evaluate_model(model, X_test, y_test)