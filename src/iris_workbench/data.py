"""Dataset loading, canonicalization, and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

import pandas as pd

CANONICAL_FEATURES = (
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
)
TARGET_COLUMN = "species"
REQUIRED_COLUMNS = (*CANONICAL_FEATURES, TARGET_COLUMN)


class DataValidationError(ValueError):
    """Raised when a dataset cannot be used safely by the workbench."""


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


_ALIASES = {
    "sepallengthcm": "sepal_length_cm",
    "sepallength": "sepal_length_cm",
    "sepalwidthcm": "sepal_width_cm",
    "sepalwidth": "sepal_width_cm",
    "petallengthcm": "petal_length_cm",
    "petallength": "petal_length_cm",
    "petalwidthcm": "petal_width_cm",
    "petalwidth": "petal_width_cm",
    "species": "species",
    "class": "species",
    "target": "species",
}


@dataclass(frozen=True)
class DatasetSummary:
    rows: int
    features: int
    class_counts: dict[str, int]
    missing_values: int
    duplicate_rows: int
    feature_ranges: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with supported Iris column variants renamed canonically."""
    rename = {column: _ALIASES.get(_key(str(column)), str(column)) for column in frame.columns}
    canonical = frame.rename(columns=rename).copy()
    if canonical.columns.duplicated().any():
        duplicates = sorted(set(canonical.columns[canonical.columns.duplicated()]))
        raise DataValidationError(f"Columns map to duplicate canonical names: {duplicates}")
    return canonical


def validate_dataset(frame: pd.DataFrame) -> DatasetSummary:
    """Validate schema and values, returning an auditable dataset summary."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {', '.join(missing_columns)}")

    if frame.empty:
        raise DataValidationError("Dataset is empty")

    if frame.loc[:, REQUIRED_COLUMNS].isna().any().any():
        counts = frame.loc[:, REQUIRED_COLUMNS].isna().sum()
        details = {column: int(count) for column, count in counts.items() if count}
        raise DataValidationError(f"Missing values found: {details}")

    for column in CANONICAL_FEATURES:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise DataValidationError(f"Column '{column}' must be numeric") from exc

    if (frame.loc[:, CANONICAL_FEATURES] <= 0).any().any():
        raise DataValidationError("All flower measurements must be positive")

    species = frame[TARGET_COLUMN].astype(str).str.strip()
    if species.eq("").any():
        raise DataValidationError("Species labels cannot be blank")
    frame[TARGET_COLUMN] = species

    class_counts = species.value_counts().sort_index()
    if len(class_counts) < 2:
        raise DataValidationError("At least two target classes are required")

    ranges = {
        column: {
            "min": float(frame[column].min()),
            "max": float(frame[column].max()),
        }
        for column in CANONICAL_FEATURES
    }
    return DatasetSummary(
        rows=len(frame),
        features=len(CANONICAL_FEATURES),
        class_counts={str(name): int(count) for name, count in class_counts.items()},
        missing_values=0,
        duplicate_rows=int(frame.loc[:, REQUIRED_COLUMNS].duplicated().sum()),
        feature_ranges=ranges,
    )


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, DatasetSummary]:
    """Load a CSV file, normalize its schema, validate it, and return both outputs."""
    source = Path(path)
    if not source.is_file():
        raise DataValidationError(f"Dataset does not exist: {source}")
    try:
        frame = pd.read_csv(source)
    except (OSError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"Could not read CSV: {source}") from exc

    frame = canonicalize_columns(frame)
    summary = validate_dataset(frame)
    return frame.loc[:, REQUIRED_COLUMNS].copy(), summary


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a validated canonical frame into features and multiclass labels."""
    return frame.loc[:, CANONICAL_FEATURES].copy(), frame[TARGET_COLUMN].copy()
