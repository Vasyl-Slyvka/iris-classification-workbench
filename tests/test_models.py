import unittest

from sklearn.pipeline import Pipeline

from iris_workbench.models import MODEL_NAMES, build_model, parameter_grid


class ModelTests(unittest.TestCase):
    def test_registry_contains_multiple_model_families(self):
        self.assertEqual(
            set(MODEL_NAMES),
            {"gaussian_nb", "knn", "logistic_regression", "random_forest", "rbf_svm"},
        )

    def test_scaled_models_are_pipelines(self):
        for name in ("knn", "logistic_regression", "rbf_svm"):
            with self.subTest(name=name):
                self.assertIsInstance(build_model(name), Pipeline)

    def test_build_model_rejects_unknown_name(self):
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            build_model("magic")

    def test_parameter_grid_is_a_defensive_copy(self):
        first = parameter_grid("knn")
        first["model__n_neighbors"].append(999)
        self.assertNotIn(999, parameter_grid("knn")["model__n_neighbors"])


if __name__ == "__main__":
    unittest.main()
