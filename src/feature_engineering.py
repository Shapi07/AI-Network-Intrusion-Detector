"""
feature_engineering.py
======================
Feature encoding, scaling, and dataset splitting for the AI Network Intrusion Detector.

Responsibilities
----------------
1. Encode categorical features (One-Hot Encoding).
2. Separate features (X) and target (y).
3. Split data into reproducible training and testing sets.
4. Scale numeric features using StandardScaler (fitted only on training data to prevent leakage).
5. Persist scaling and feature name artifacts for future inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURE_NAMES_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    SCALER_PATH,
    TEST_SIZE,
    TARGET_COLUMN,
    configure_logging,
)
from src.utils import ensure_dir, timeit

logger = configure_logging(__name__)


@timeit
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features using one-hot encoding.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe containing both numeric and object/categorical columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with categorical columns one-hot encoded.
    """
    logger.info("Encoding categorical features...")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if TARGET_COLUMN in cat_cols:
        cat_cols.remove(TARGET_COLUMN)

    if not cat_cols:
        logger.info("No categorical columns found to encode.")
        return df

    logger.info("Found %d categorical column(s): %s", len(cat_cols), cat_cols)
    
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    logger.info("Categorical encoding complete. New shape: %s", df_encoded.shape)
    return df_encoded


@timeit
def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataframe into feature matrix (X) and target vector (y).

    Parameters
    ----------
    df : pd.DataFrame
        Fully cleaned and encoded dataframe.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix X and target vector y.

    Raises
    ------
    KeyError
        If TARGET_COLUMN is missing from the dataframe.
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")

    logger.info("Splitting features and target column: '%s'", TARGET_COLUMN)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


@timeit
def scale_numeric_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    save_artifacts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale numeric feature columns using StandardScaler fitted only on training data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Testing feature matrix.
    save_artifacts : bool, optional
        Whether to persist the fitted scaler and feature names to disk.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Scaled training and testing feature matrices as DataFrames.
    """
    logger.info("Scaling numeric features with StandardScaler...")
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    if save_artifacts:
        ensure_dir(MODELS_DIR)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(list(X_train.columns), FEATURE_NAMES_PATH)
        logger.info("Saved scaler artifact → %s", SCALER_PATH)
        logger.info("Saved feature names artifact → %s", FEATURE_NAMES_PATH)

    return X_train_scaled, X_test_scaled


@timeit
def prepare_data(
    df: pd.DataFrame,
    *,
    save_artifacts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    End-to-end feature engineering and train/test split pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe from Phase 3.
    save_artifacts : bool, optional
        Whether to save preprocessing artifacts.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test.
    """
    logger.info("Starting feature engineering pipeline...")

    df_encoded = encode_categorical(df)
    X, y = split_features_target(df_encoded)

    logger.info(
        "Splitting dataset: test_size=%.2f, random_state=%d",
        TEST_SIZE,
        RANDOM_STATE,
    )
    
    stratify_target = y if y.nunique() > 1 and y.value_counts().min() > 1 else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_target,
    )

    X_train_scaled, X_test_scaled = scale_numeric_features(
        X_train, X_test, save_artifacts=save_artifacts
    )

    logger.info(
        "Feature engineering complete. X_train: %s, X_test: %s",
        X_train_scaled.shape,
        X_test_scaled.shape,
    )

    return X_train_scaled, X_test_scaled, y_train, y_test