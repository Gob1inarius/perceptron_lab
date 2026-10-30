from perceptron.model import Perceptron


def test_activate_with_zero() -> None:
    p = Perceptron(n_features=2)
    assert p.activate(0) == 1


def test_activate_with_negative() -> None:
    p = Perceptron(n_features=2)
    assert p.activate(-1) == 0


def test_init_weights_length() -> None:
    p = Perceptron(n_features=2)
    assert len(p.w) == 2
