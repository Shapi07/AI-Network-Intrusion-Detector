"""
train.py
========
Model training, evaluation comparison, and selection for the AI Network Intrusion Detector.

Responsibilities
----------------
1. Initialize machine learning models (Random Forest, Logistic Regression, Decision Tree)
   using hyperparameters defined in `config.py`[cite: 9].
2. Train models on the scaled training dataset (`X_train`, `y_train`).
3. Evaluate and compare models on the test set (`X_test`, `y_test`) using Accuracy and F1-score.
4. Select the best-performing model and persist it to disk as `models/best_model.joblib`[cite: 9].
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    DECISION_TREE_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
    MODEL_PATH,
    RANDOM_FOREST_PARAMS,
    configure_logging,
)
from src.utils import ensure_dir, timeit

logger = configure_logging(__name__)


def get_models() -> dict[str, Any]:
    """
    Initialize and return the dictionary of models to train using configuration parameters[cite: 9].

    Returns
    -------
    dict[str, Any]
        Mapping of model names to un-fitted estimator instances.
    """
    logger.info("Initializing models with parameters from config[cite: 9]...")
    return {
        "Random Forest": RandomForestClassifier(**RANDOM_FOREST_PARAMS),
        "Logistic Regression": LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
        "Decision Tree": DecisionTreeClassifier(**DECISION_TREE_PARAMS),
    }


@timeit
def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Train a single machine learning model.

    Parameters
    ----------
    model : Any
        Scikit-learn classifier instance.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.

    Returns
    -------
    Any
        Fitted model instance.
    """
    model_name = model.__class__.__name__
    logger.info("Training %s...", model_name)
    model.fit(X_train, y_train)
    logger.info("Finished training %s.", model_name)
    return model


@timeit
def evaluate_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[float, float]:
    """
    Evaluate a trained model on test data using accuracy and F1 score.

    Parameters
    ----------
    model : Any
        Fitted classifier.
    X_test : pd.DataFrame
        Testing feature matrix.
    y_test : pd.Series
        Testing target vector.

    Returns
    -------
    tuple[float, float]
        (accuracy, f1_score).
    """
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    return float(acc), float(f1)


@timeit
def train_and_select_best_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    save_best: bool = True,
) -> tuple[str, Any, dict[str, dict[str, float]]]:
    """
    Train all configured models, evaluate their performance, select the best one
    based on F1-score, and optionally persist it to disk[cite: 9].

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Testing features.
    y_train : pd.Series
        Training target.
    y_test : pd.Series
        Testing target.
    save_best : bool, optional
        Whether to save the best model artifact to disk.

    Returns
    -------
    tuple[str, Any, dict[str, dict[str, float]]]
        * Name of the best model.
        * Fitted best model instance.
        * Performance metrics summary for all models.
    """
    models = get_models()
    results: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Any] = {}

    for name, model in models.items():
        logger.info("--- Pipeline for: %s ---", name)
        fitted = train_model(model, X_train, y_train)
        fitted_models[name] = fitted

        acc, f1 = evaluate_model(fitted, X_test, y_test)
        results[name] = {"accuracy": acc, "f1_score": f1}
        logger.info("%s -> Accuracy: %.4f | F1-Score: %.4f", name, acc, f1)

    # Select the best model based on weighted F1-score
    best_model_name = max(results, key=lambda k: results[k]["f1_score"])
    best_model = fitted_models[best_model_name]

    logger.info(
        "🏆 Best model selected: %s (F1-Score: %.4f)",
        best_model_name,
        results[best_model_name]["f1_score"],
    )

    if save_best:
        ensure_dir(MODEL_PATH.parent)
        joblib.dump(best_model, MODEL_PATH)
        logger.info("Saved best model artifact -> %s[cite: 9]", MODEL_PATH)

    return best_model_name, best_model, results