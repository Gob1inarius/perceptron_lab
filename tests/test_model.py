import numpy as np

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


def test_predict() -> None:
    p = Perceptron(n_features=2)
    p.w = np.array([1.0, 1.0])
    p.bias = -1.5
    assert p.predict(np.array([1.0, 1.0])) == 1
    assert p.predict(np.array([0.0, 1.0])) == 0
    assert p.predict(np.array([1.0, 0.0])) == 0
    assert p.predict(np.array([0.0, 0.0])) == 0


def test_train_learn() -> None:
    p = Perceptron(n_features=2)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 0, 1])
    p.train(X, y)
    assert p.predict(np.array([1.0, 1.0])) == 1
    assert p.predict(np.array([0.0, 1.0])) == 0
    assert p.predict(np.array([1.0, 0.0])) == 0
    assert p.predict(np.array([0.0, 0.0])) == 0

def test_errors_len_equals_epochs() -> None:
    p = Perceptron(n_features=2)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 0, 1])
    errors = p.train(X, y)
    assert len(errors) == p.epochs

def test_score_after_training_is_one() -> None:
    p = Perceptron(n_features=2)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 0, 1])
    p.train(X, y)
    assert p.score(X, y) == 1.0

def test_score_before_training_is_quarter() -> None:
    p = Perceptron(n_features=2)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 0, 1])
    assert p.score(X, y) == 0.25