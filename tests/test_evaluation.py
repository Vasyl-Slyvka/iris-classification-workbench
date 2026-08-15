from pathlib import Path
import unittest

from iris_workbench.data import load_dataset
from iris_workbench.evaluation import compare_models, evaluate_holdout, tune_model

ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, _ = load_dataset(ROOT / "data" / "iris.csv")

    def test_comparison_is_sorted_and_reproducible(self):
        options = {
            "models": ["knn", "logistic_regression"],
            "folds": 3,
            "repeats": 1,
            "seed": 7,
        }
        first = compare_models(self.frame, **options)
        second = compare_models(self.frame, **options)
        self.assertEqual(first["model"].tolist(), second["model"].tolist())
        for column in ("accuracy_mean", "accuracy_std", "balanced_accuracy_mean", "f1_macro_mean"):
            self.assertEqual(first[column].tolist(), second[column].tolist())
        self.assertTrue(first["f1_macro_mean"].is_monotonic_decreasing)

    def test_comparison_metrics_stay_in_unit_interval(self):
        results = compare_models(
            self.frame, models=["gaussian_nb", "knn"], folds=3, repeats=1
        )
        for column in ("accuracy_mean", "accuracy_std", "balanced_accuracy_mean", "f1_macro_mean"):
            self.assertTrue(results[column].between(0, 1).all())

    def test_holdout_diagnostics_are_reproducible(self):
        first = evaluate_holdout(self.frame, "knn", seed=42)
        second = evaluate_holdout(self.frame, "knn", seed=42)
        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["confusion_matrix"], second["confusion_matrix"])
        self.assertEqual(first["test_rows"], 30)
        self.assertEqual(len(first["confusion_matrix"]), 3)

    def test_tuning_returns_best_estimator(self):
        search = tune_model(self.frame, "logistic_regression", folds=3, seed=42)
        self.assertTrue(hasattr(search, "best_estimator_"))
        self.assertIn("model__C", search.best_params_)
        self.assertGreater(search.best_score_, 0.8)

    def test_invalid_folds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "folds"):
            compare_models(self.frame, models=["knn"], folds=1)


if __name__ == "__main__":
    unittest.main()
