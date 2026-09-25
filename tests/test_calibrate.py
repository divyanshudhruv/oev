import torch

from oev.calibrate import apply_temperature, fit_temperature_from_logits


def overconfident_logits():
    return torch.tensor([[5.0, 0.0, 0.0], [4.0, 0.5, 0.0], [6.0, 1.0, 0.0], [3.0, 2.0, 0.0]])


def wrong_labels():
    return torch.tensor([1, 0, 1, 0])


def test_argmax_invariant():
    scaled = apply_temperature(overconfident_logits(), 4.0)
    assert scaled.argmax(dim=1).tolist() == overconfident_logits().argmax(dim=1).tolist()


def test_higher_temperature_lowers_confidence():
    probs_raw = torch.softmax(overconfident_logits(), dim=1).max(dim=1).values
    probs_cal = torch.softmax(apply_temperature(overconfident_logits(), 4.0), dim=1).max(dim=1).values
    assert (probs_cal < probs_raw).all()


def test_fit_raises_temperature_when_overconfident():
    t = fit_temperature_from_logits(overconfident_logits(), wrong_labels())
    assert t > 1.0
