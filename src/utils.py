"""
utils.py
========
Shared utility helpers used across all modules.

Responsibilities
----------------
* Timing decorator for profiling long-running steps.
* Safe directory creation.
* JSON serialisation / deserialisation.
* Dataframe memory-usage reporter.
* Seed-setting for full reproducibility.
"""

from __future__ import annotations

import functools
import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.config import configure_logging

logger = configure_logging(__name__)


# ──────────────────────────────────────────────────────────────
# Timing decorator
# ──────────────────────────────────────────────────────────────

def timeit(func: Callable) -> Callable:
    """
    Decorator that logs the wall-clock execution time of *func*.

    Example
    -------
    >>> @timeit
    ... def slow_function():
    ...     time.sleep(1)
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(
            "⏱  %s finished in %.2f seconds", func.__qualname__, elapsed
        )
        return result

    return wrapper


# ──────────────────────────────────────────────────────────────
# File system helpers
# ──────────────────────────────────────────────────────────────

def ensure_dir(path: Path | str) -> Path:
    """
    Create *path* (and all parents) if it does not already exist.

    Parameters
    ----------
    path : Path | str
        Directory path to create.

    Returns
    -------
    Path
        The resolved, guaranteed-to-exist directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ──────────────────────────────────────────────────────────────
# JSON helpers
# ──────────────────────────────────────────────────────────────

def save_json(data: dict[str, Any], path: Path | str) -> None:
    """
    Serialise *data* to a JSON file at *path*.

    Uses ``indent=4`` for human-readable output.  Floats are
    rounded to 6 decimal places to keep the file concise.

    Parameters
    ----------
    data : dict[str, Any]
        Serialisable dictionary.
    path : Path | str
        Destination file path.
    """
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=4, default=_json_serialiser)

    logger.info("📄 Saved JSON → %s", path)


def load_json(path: Path | str) -> dict[str, Any]:
    """
    Load a JSON file and return the parsed dictionary.

    Parameters
    ----------
    path : Path | str
        Source file path.

    Returns
    -------
    dict[str, Any]
        Parsed JSON content.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _json_serialiser(obj: Any) -> Any:
    """
    Custom JSON serialiser for types not handled by the stdlib.

    Handles: ``numpy`` scalar types and ``Path`` objects.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 6)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────

def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy to ensure reproducibility.

    Parameters
    ----------
    seed : int
        The integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.debug("🌱 Global random seed set to %d", seed)


# ──────────────────────────────────────────────────────────────
# DataFrame diagnostics
# ──────────────────────────────────────────────────────────────

def dataframe_memory_report(df: "pd.DataFrame") -> str:  # noqa: F821
    """
    Return a human-readable string showing the DataFrame's memory footprint.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to inspect.

    Returns
    -------
    str
        Memory usage summary string.
    """
    mem_bytes = df.memory_usage(deep=True).sum()
    mem_mb = mem_bytes / (1024 ** 2)
    return (
        f"DataFrame: {df.shape[0]:,} rows × {df.shape[1]} columns | "
        f"Memory: {mem_mb:.2f} MB"
    )
