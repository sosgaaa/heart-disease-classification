import contextlib
import io
import unittest

import numpy as np

from evaluate import metrics, run_study
from main import stratified_folds


class EvaluationTests(unittest.TestCase):
    def test_fixed_class_macro_f1_includes_unpredicted_classes(self):
        result = metrics(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]), np.array([0, 1]))
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)
        self.assertEqual(result["confusion_matrix"], [[2, 0], [2, 0]])
        self.assertEqual(result["class_recall"], [1.0, 0.0])

    def test_folds_reject_missing_class_support(self):
        with self.assertRaises(ValueError):
            list(stratified_folds(np.array([0, 0, 0, 1]), 2, 42))

    def test_test_data_and_labels_cannot_change_selection_or_nested_scores(self):
        rng = np.random.default_rng(17)
        training = rng.normal(size=(24, 2))
        labels = np.array([0] * 12 + [1] * 12)
        training[12:] += 2
        grids = {
            "knn": [{"k": 1, "features": 1}, {"k": 3, "features": 2}],
            "kmeans": [{"k": 2, "features": 1}, {"k": 3, "features": 2}],
            "logistic_regression": [{"lr": 0.01, "max_iters": 20, "reg": 0.001},
                                    {"lr": 0.1, "max_iters": 30, "reg": 0.001}],
        }
        with contextlib.redirect_stdout(io.StringIO()):
            first = run_study(training, labels, np.zeros((4, 2)), np.array([0, 0, 1, 1]), grids, 3, 2)
            changed = run_study(training, labels, np.full((4, 2), 1000.0), np.array([1, 1, 1, 0]), grids, 3, 2)
        for method in grids:
            for field in ("selected_params", "search", "nested_cv", "nested_folds", "out_of_fold"):
                self.assertEqual(first["methods"][method][field], changed["methods"][method][field])
            folds = first["methods"][method]["nested_folds"]
            indices = [index for fold in folds for index in fold["validation_indices"]]
            self.assertEqual(sorted(indices), list(range(24)))
            self.assertEqual(np.sum(first["methods"][method]["out_of_fold"]["confusion_matrix"]), 24)


if __name__ == "__main__":
    unittest.main()
