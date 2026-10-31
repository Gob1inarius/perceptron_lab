"""Perceptron model."""

import numpy as np
import numpy.typing as npt


class Perceptron:
    """Single-layer neural network for classification."""

    def __init__(
        self, n_features: int, learning_rate: float = 0.1, epochs: int = 50
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w: npt.NDArray[np.float64] = np.zeros(n_features)
        self.bias = 0.0

    def activate(self, z: float) -> int:
        """Return 1 if z is greater than or equal to 0, otherwise 0."""
        if z >= 0:
            return 1
        else:
            return 0

    def predict(self, x: npt.NDArray[np.float64]) -> int:
        """Compute sum w1*x1 + w2*x2 + ... + bias and passes it through activate."""
        z = np.dot(self.w, x) + self.bias
        return self.activate(z)

    def train(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.int_]) -> list[int]:
        """Fit weights on X, y; return the number of misclassified samples per epoch."""
        errors: list[int] = []
        for _ in range(self.epochs):
            errors_counter = 0
            for xi, target in zip(X, y, strict = True):
                pred = self.predict(xi)
                if target != pred:
                    errors_counter += 1
                self.w += self.learning_rate * (target - pred) * xi
                self.bias += self.learning_rate * (target - pred)
            errors.append(errors_counter)
        return errors

    def score(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.int_]) -> float:
        """Compute the proportion of correct answers"""
        correct_num = 0
        for xi, target in zip(X, y, strict=True):
            if target == self.predict(xi):
                correct_num += 1
        return correct_num / len(y)