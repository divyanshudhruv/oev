from oev.evaluate import ece, majority_baseline


def test_ece_perfect_is_zero():
    assert ece([0.99] * 100, [1] * 100) < 0.02


def test_ece_overconfident_wrong():
    assert ece([0.9] * 100, [0] * 100) > 0.8


def test_majority_baseline():
    assert abs(majority_baseline(["a", "a", "b"]) - 2 / 3) < 1e-9
