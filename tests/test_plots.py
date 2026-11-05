from pathlib import Path

from matplotlib.figure import Figure

from perceptron.data import make_clouds
from perceptron.model import Perceptron
from perceptron.plots import plot_decision_boundary, plot_errors


def test_plot_errors_returns_figure() -> None:
    fig = plot_errors([5, 2, 0])
    assert isinstance(fig, Figure)


def test_plot_errors_can_be_saved(tmp_path: Path) -> None:
    fig = plot_errors([5, 2, 0])
    file = tmp_path / "errors.png"
    fig.savefig(file)
    assert file.exists()


def test_untrained_model_has_no_boundary_line() -> None:
    p = Perceptron(n_features=2)
    X, y = make_clouds(50)
    fig = plot_decision_boundary(p, X, y)
    assert len(fig.axes[0].lines) == 0


def test_trained_model_has_boundary_line() -> None:
    p = Perceptron(n_features=2)
    X, y = make_clouds(50)
    p.train(X, y)
    fig = plot_decision_boundary(p, X, y)
    assert len(fig.axes[0].lines) == 1


def test_plot_decision_boundary_can_be_saved(tmp_path: Path) -> None:
    p = Perceptron(n_features=2)
    X, y = make_clouds(50)
    p.train(X, y)
    fig = plot_decision_boundary(p, X, y)
    file = tmp_path / "boundary.png"
    fig.savefig(file)
    assert file.exists()


def test_plot_decision_boundary_untrained_model() -> None:
    p = Perceptron(n_features=2)
    X, y = make_clouds(50)
    fig = plot_decision_boundary(p, X, y)
    assert isinstance(fig, Figure)
