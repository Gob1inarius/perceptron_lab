from perceptron.model import activate


def test_activate_with_zero() -> None:
    assert activate(0) == 1


def test_activate_with_negative() -> None:
    assert activate(-1) == 0
