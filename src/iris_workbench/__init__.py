"""Reproducible classical-ML experiments on the Iris dataset."""

from .data import CANONICAL_FEATURES, TARGET_COLUMN, DataValidationError, load_dataset
from .models import MODEL_NAMES, build_model

__all__ = [
    "CANONICAL_FEATURES",
    "TARGET_COLUMN",
    "DataValidationError",
    "MODEL_NAMES",
    "build_model",
    "load_dataset",
]

__version__ = "1.0.0"
