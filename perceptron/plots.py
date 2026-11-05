"""Plots for Perceptron."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

from perceptron.model import Perceptron


def plot_decision_boundary(
    model: Perceptron, X: npt.NDArray[np.float64], y: npt.NDArray[np.int_]
) -> Figure:
    """Plot both classes and line found by Perceptron."""
    fig, ax = plt.subplots()
    ax.scatter(X[y == 0, 0], X[y == 0, 1], label="class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], label="class 1")

    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    if model.w[1] != 0:
        y_at_min = -(model.w[0] * x_min + model.bias) / model.w[1]
        y_at_max = -(model.w[0] * x_max + model.bias) / model.w[1]
        ax.plot([x_min, x_max], [y_at_min, y_at_max], color="black")
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Decision boundary")
    ax.legend()
    return fig


def plot_errors(errors: list[int]) -> Figure:
    """Plot number of misclassified samples per epoch."""
    fig, ax = plt.subplots()
    epochs = list(range(1, len(errors) + 1))
    ax.plot(epochs, errors)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Errors")
    ax.set_title("Training errors")
    return fig
