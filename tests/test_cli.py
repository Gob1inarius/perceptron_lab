import pytest

from perceptron.cli import parse_args


def test_default_values_are_correct() -> None:
    args = parse_args([])
    assert args.samples == 200
    assert args.lr == 0.1
    assert args.epochs == 50


def test_user_values_are_correct() -> None:
    args = parse_args(["--samples", "100", "--epochs", "10"])
    assert args.samples == 100
    assert args.epochs == 10


def test_samples_too_small() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--samples", "1"])


def test_learning_rate_too_small() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--lr", "0"])


def test_epochs_too_small() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--epochs", "0"])
