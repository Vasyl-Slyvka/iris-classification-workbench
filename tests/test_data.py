from pathlib import Path
import tempfile
import unittest

import pandas as pd

from iris_workbench.data import (
    CANONICAL_FEATURES,
    DataValidationError,
    load_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "iris.csv"


class DataTests(unittest.TestCase):
    def test_loads_and_canonicalizes_repository_dataset(self):
        frame, summary = load_dataset(DATA)
        self.assertEqual(list(frame.columns[:4]), list(CANONICAL_FEATURES))
        self.assertEqual(summary.rows, 150)
        self.assertEqual(summary.features, 4)
        self.assertEqual(set(summary.class_counts.values()), {50})
        self.assertEqual(summary.missing_values, 0)
        self.assertEqual(summary.duplicate_rows, 3)

    def test_accepts_common_column_aliases(self):
        source = pd.DataFrame(
            {
                "Sepal Length": [5.1, 6.2, 6.8, 5.0],
                "Sepal Width": [3.5, 2.8, 3.0, 3.4],
                "Petal Length": [1.4, 4.8, 5.5, 1.5],
                "Petal Width": [0.2, 1.8, 2.1, 0.2],
                "Class": ["a", "b", "c", "a"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aliases.csv"
            source.to_csv(path, index=False)
            frame, _ = load_dataset(path)
        self.assertEqual(list(frame.columns), [*CANONICAL_FEATURES, "species"])

    def test_rejects_missing_column(self):
        source = pd.read_csv(DATA).drop(columns=["PetalWidthcm"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.csv"
            source.to_csv(path, index=False)
            with self.assertRaisesRegex(DataValidationError, "petal_width_cm"):
                load_dataset(path)

    def test_rejects_missing_value(self):
        source = pd.read_csv(DATA)
        source.loc[0, "SepalLengthCm"] = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-value.csv"
            source.to_csv(path, index=False)
            with self.assertRaisesRegex(DataValidationError, "Missing values"):
                load_dataset(path)

    def test_rejects_non_numeric_feature(self):
        source = pd.read_csv(DATA)
        source["SepalLengthCm"] = source["SepalLengthCm"].astype(object)
        source.loc[0, "SepalLengthCm"] = "not-a-number"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            source.to_csv(path, index=False)
            with self.assertRaisesRegex(DataValidationError, "must be numeric"):
                load_dataset(path)

    def test_rejects_non_positive_measurement(self):
        source = pd.read_csv(DATA)
        source.loc[0, "SepalLengthCm"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zero.csv"
            source.to_csv(path, index=False)
            with self.assertRaisesRegex(DataValidationError, "must be positive"):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
