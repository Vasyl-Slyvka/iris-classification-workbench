from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import tempfile
import unittest

from iris_workbench.cli import main

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "iris.csv"


class CliTests(unittest.TestCase):
    def test_validate_writes_json(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "summary.json"
            with redirect_stdout(StringIO()):
                exit_code = main(
                    ["validate", "--data", str(DATA), "--json", str(destination)]
                )
            payload = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["rows"], 150)

    def test_compare_writes_csv_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "comparison.csv"
            json_path = Path(directory) / "comparison.json"
            with redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        "compare",
                        "--data",
                        str(DATA),
                        "--models",
                        "knn",
                        "logistic_regression",
                        "--folds",
                        "3",
                        "--repeats",
                        "1",
                        "--csv",
                        str(csv_path),
                        "--json",
                        str(json_path),
                    ]
                )
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            csv_was_written = csv_path.is_file()
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(payload), 2)
        self.assertTrue(csv_was_written)

    def test_missing_dataset_returns_controlled_error(self):
        with redirect_stdout(StringIO()) as stream:
            exit_code = main(["validate", "--data", "definitely-missing.csv"])
        self.assertEqual(exit_code, 2)
        self.assertIn("error:", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
