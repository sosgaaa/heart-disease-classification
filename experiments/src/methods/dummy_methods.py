"""Uniform-random classifier used as an exploratory baseline."""

import numpy as np

from ..utils import get_n_classes


class DummyClassifier:
    """Predict labels uniformly at random from the fitted class range."""

    def __init__(self, arg1, arg2=0):
        """Retain compatibility arguments used by the experiment entry point."""
        self.arg1 = arg1
        self.arg2 = arg2

    def random_predict(self, C, N):
        """Draw N class indices uniformly from 0 to C - 1."""
        return np.random.randint(low=0, high=C, size=N)

    def fit(self, training_data, training_labels):
        """Record the feature and class counts, then predict training labels."""
        self.D, self.C = training_data.shape[1], get_n_classes(training_labels)
        return self.predict(training_data)

    def predict(self, test_data):
        """Return one random class index per evaluation sample."""
        return self.random_predict(self.C, test_data.shape[0])
