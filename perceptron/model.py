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
