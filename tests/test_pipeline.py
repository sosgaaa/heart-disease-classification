import unittest
from pathlib import Path

import numpy as np

from main import load_dataset, preprocess, stratified_folds
from src.methods.kmeans import KMeans
from src.methods.knn import KNN
from src.methods.logistic_regression import LogisticRegression
from src.utils import append_bias_term


class PipelineTests(unittest.TestCase):
    def test_bundled_dataset_shapes(self):
        xtrain, xtest, ytrain, ytest = load_dataset(Path(__file__).parents[1] / "features.npz")
        self.assertEqual(xtrain.shape, (237, 13))
        self.assertEqual(xtest.shape, (60, 13))
        self.assertEqual(ytrain.shape, (237,))
        self.assertEqual(ytest.shape, (60,))

    def test_evaluation_rows_do_not_change_fitted_scaler(self):
        train = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
        eval_a = np.array([[10.0, 20.0]])
        eval_b = np.array([[1000.0, 2000.0]])
        fitted_a, _ = preprocess(train, eval_a, "knn")
        fitted_b, _ = preprocess(train, eval_b, "knn")
        np.testing.assert_array_equal(fitted_a, fitted_b)

    def test_folds_cover_each_sample_once(self):
        labels = np.array([0] * 12 + [1] * 12)
        folds = list(stratified_folds(labels, 4, 42))
        validation = np.concatenate([valid for _, valid in folds])
        np.testing.assert_array_equal(np.sort(validation), np.arange(len(labels)))
        for train, valid in folds:
            self.assertEqual(len(np.intersect1d(train, valid)), 0)

    def test_classifiers_predict_known_clusters(self):
        training = np.array([[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]])
        labels = np.array([0, 0, 0, 1, 1, 1])
        evaluation = np.array([[0.05], [10.05]])
        for model in (KNN(k=3), KMeans(k=2, random_state=42)):
            model.fit(training, labels)
            np.testing.assert_array_equal(model.predict(evaluation), [0, 1])
        logistic = LogisticRegression(lr=0.1, max_iters=1000)
        logistic.fit(append_bias_term(training), labels)
        np.testing.assert_array_equal(logistic.predict(append_bias_term(evaluation)), [0, 1])


if __name__ == "__main__":
    unittest.main()
