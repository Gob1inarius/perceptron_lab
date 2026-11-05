import numpy as np

from perceptron.data import load_iris_binary, make_clouds
from perceptron.model import Perceptron


def test_cloud_shape_for_even() -> None:
    X, y = make_clouds(200)
    assert X.shape == (200, 2)
    assert len(y) == 200


def test_cloud_shape_for_odd() -> None:
    X, y = make_clouds(201)
    assert X.shape == (201, 2)
    assert len(y) == 201


def test_two_clouds_exist() -> None:
    _, y = make_clouds(200)
    assert set(y.tolist()) == {0, 1}


def test_same_seeds_give_same_results() -> None:
    X1, _ = make_clouds(200, seed=123)
    X2, _ = make_clouds(200, seed=123)
    assert np.array_equal(X1, X2)


def test_different_seeds_give_defferent_results() -> None:
    X1, _ = make_clouds(200, seed=123)
    X2, _ = make_clouds(200, seed=321)
    assert not np.array_equal(X1, X2)


def test_for_seperated_data_score_is_one() -> None:
    p = Perceptron(n_features=2)
    X, y = make_clouds(200)
    p.train(X, y)
    assert p.score(X, y) == 1.0


def test_iris_shape() -> None:
    X, y = load_iris_binary()
    assert X.shape == (150, 2)
    assert len(y) == 150


def test_iris_balance() -> None:
    _, y = load_iris_binary()
    assert int(y.sum()) == 50


def test_same_iris_seeds_give_same_results() -> None:
    X1, _ = load_iris_binary(seed=123)
    X2, _ = load_iris_binary(seed=123)
    assert np.array_equal(X1, X2)


def test_different_iris_seeds_give_defferent_results() -> None:
    X1, _ = load_iris_binary(seed=123)
    X2, _ = load_iris_binary(seed=321)
    assert not np.array_equal(X1, X2)


def test_iris_data_is_shuffled() -> None:
    _, y = load_iris_binary()
    assert set(y[:50].tolist()) == {0, 1}


def test_training_on_iris_score_is_one() -> None:
    p = Perceptron(n_features=2)
    X, y = load_iris_binary()
    p.train(X, y)
    assert p.score(X, y) == 1.0
