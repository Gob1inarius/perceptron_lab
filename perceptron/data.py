"""Data for perceptron."""

import numpy as np
import numpy.typing as npt


def make_clouds(
    n_samples: int, seed: int = 123
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]]:
    """Generate two clouds."""
    rng = np.random.default_rng(seed)
    n0 = n_samples // 2
    n1 = n_samples - n0
    points0 = rng.normal(loc=[-2.0, -2.0], scale=0.5, size=(n0, 2))
    points1 = rng.normal(loc=[2.0, 2.0], scale=0.5, size=(n1, 2))
    answers0 = np.zeros(n0, dtype=np.int_)
    answers1 = np.ones(n1, dtype=np.int_)
    X = np.vstack([points0, points1])
    y = np.concatenate([answers0, answers1])
    index = rng.permutation(n_samples)
    X = X[index]
    y = y[index]
    return X, y
