import numpy as np

from perceptron.data import make_clouds
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


def test_integration() -> None:
    p = Perceptron(n_features=2)
    X, y = make_clouds(200)
    p.train(X, y)
    assert p.score(X, y) == 1.0
