"""Command line interface for perceptron."""

import argparse

import matplotlib.pyplot as plt

from perceptron.data import load_iris_binary, make_clouds
from perceptron.model import Perceptron
from perceptron.plots import plot_decision_boundary, plot_errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read arguments of command line."""
    parser = argparse.ArgumentParser(
        description="Train a perceptron on two synthetic 2D clusters and print its accuracy."
    )
    parser.add_argument("--samples", type=int, default=200, help="number of points")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="build decision boundary and error plots, save them as PNG and show them",
    )
    parser.add_argument(
        "--dataset",
        choices=["clouds", "iris"],
        default="clouds",
        help="dataset to use; --samples is ignored for iris",
    )

    args = parser.parse_args(argv)

    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.lr <= 0:
        parser.error("--lr must be more than 0")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dataset == "clouds":
        X, y = make_clouds(args.samples)
    else:
        X, y = load_iris_binary()
    p = Perceptron(n_features=X.shape[1], learning_rate=args.lr, epochs=args.epochs)
    errors = p.train(X, y)
    print(f"Accuracy: {p.score(X, y):.2f}")
    if args.plot:
        plot_decision_boundary(p, X, y).savefig("boundary.png")
        plot_errors(errors).savefig("errors.png")
        plt.show()


if __name__ == "__main__":
    main()
