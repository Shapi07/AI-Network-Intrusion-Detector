"""
config.py
=========
Central configuration for the AI Network Intrusion Detector.

All file paths, model hyperparameters, feature lists, and
logging settings are defined here so every module has a
single source of truth.  Import this module rather than
hard-coding values anywhere else in the project.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Project root — everything is relative to this
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────────────────────
# Directory layout
# ──────────────────────────────────────────────────────────────
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# Ensure runtime directories exist (idempotent)
for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Data files
# ──────────────────────────────────────────────────────────────
# Default dataset name (UNSW-NB15 CSV).  The user can override
# this by setting the AINID_DATASET env-var or passing a path
# directly to the loader functions.
DEFAULT_DATASET_FILENAME: str = os.getenv(
    "AINID_DATASET", "UNSW_NB15_training-set.csv"
)
DEFAULT_DATASET_PATH: Path = RAW_DATA_DIR / DEFAULT_DATASET_FILENAME

PROCESSED_DATASET_PATH: Path = PROCESSED_DATA_DIR / "processed_data.csv"

# ──────────────────────────────────────────────────────────────
# Model persistence
# ──────────────────────────────────────────────────────────────
MODEL_FILENAME: str = "best_model.joblib"
SCALER_FILENAME: str = "scaler.joblib"
LABEL_ENCODER_FILENAME: str = "label_encoder.joblib"
FEATURE_NAMES_FILENAME: str = "feature_names.joblib"

MODEL_PATH: Path = MODELS_DIR / MODEL_FILENAME
SCALER_PATH: Path = MODELS_DIR / SCALER_FILENAME
LABEL_ENCODER_PATH: Path = MODELS_DIR / LABEL_ENCODER_FILENAME
FEATURE_NAMES_PATH: Path = MODELS_DIR / FEATURE_NAMES_FILENAME

# ──────────────────────────────────────────────────────────────
# Target column
# ──────────────────────────────────────────────────────────────
# UNSW-NB15 uses "label" (0 = normal, 1 = attack).
# CIC-IDS2017 uses " Label" (with a leading space — stripped
# during preprocessing).
TARGET_COLUMN: str = os.getenv("AINID_TARGET_COLUMN", "label")

# Human-readable class names used in reports and charts
CLASS_NAMES: list[str] = ["Normal", "Attack"]

# ──────────────────────────────────────────────────────────────
# UNSW-NB15 — columns to drop
# ──────────────────────────────────────────────────────────────
# These columns are identifiers or have too many categories to
# be useful without careful encoding.  Dropping them keeps the
# feature space clean for the first training run.
COLUMNS_TO_DROP: list[str] = [
    "id",         # row identifier
    "attack_cat", # attack category (multi-class; we do binary)
]

# ──────────────────────────────────────────────────────────────
# Train / test split
# ──────────────────────────────────────────────────────────────
TEST_SIZE: float = 0.20          # 80 % train, 20 % test
RANDOM_STATE: int = 42           # reproducibility seed

# ──────────────────────────────────────────────────────────────
# Model hyperparameters
# ──────────────────────────────────────────────────────────────
RANDOM_FOREST_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": None,           # grow until pure leaves
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "n_jobs": -1,                # use all CPU cores
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",  # handles class imbalance
}

LOGISTIC_REGRESSION_PARAMS: dict = {
    "max_iter": 1000,
    "solver": "lbfgs",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
}

DECISION_TREE_PARAMS: dict = {
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
}

# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────
METRICS_REPORT_PATH: Path = MODELS_DIR / "metrics_report.json"

# Number of top features to display in feature-importance charts
TOP_N_FEATURES: int = 20

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
LOG_LEVEL: int = logging.INFO
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_FILE: Path = LOGS_DIR / "ainid.log"


def configure_logging(name: str = "ainid") -> logging.Logger:
    """
    Create and return a logger with both console and file handlers.

    Parameters
    ----------
    name : str
        Logger name — typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
