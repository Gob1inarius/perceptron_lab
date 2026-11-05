"""Command line interface for perceptron."""

import argparse

from perceptron.data import make_clouds
from perceptron.model import Perceptron


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read arguments of command line."""
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--samples", type=int, default=200, help="number of points")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")

    args = parser.parse_args(argv)

    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.lr <= 0:
        parser.error("--lr must be more than 0")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    return args


def main() -> None:
    args = parse_args()
    X, y = make_clouds(args.samples)
    p = Perceptron(n_features=2, learning_rate=args.lr, epochs=args.epochs)
    p.train(X, y)
    print(f"Accuracy: {p.score(X, y):.2f}")


if __name__ == "__main__":
    main()
