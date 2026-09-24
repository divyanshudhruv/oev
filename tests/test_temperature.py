import torch

from oev.infer import OEV


class _StubModel:
    # minimal stand-in: _probs only touches .cfg and calls the model
    class cfg:
        max_len = 128
        backbone = None

    def __call__(self, ids, pad, anchors):
        # fixed raw logits: option 0 strongest
        base = torch.tensor([2.0, 1.0, 0.0])
        B, A = anchors.shape
        return base.expand(B, A).clamp(max=2.0)[:, :A] + torch.zeros(B, A)


def _make_oev(n_options=3):
    oev = OEV.__new__(OEV)
    oev.model = _StubModel()
    oev.device = "cpu"
    oev.temperature = 1.0
    oev.max_len = 128
    oev.packer = None
    return oev


def test_temperature_changes_confidence_not_argmax():
    oev = _make_oev()
    q = {"type": "choice", "options": ["a", "b", "c"], "answer": "a"}
    pq = oev._question("q", q)

    oev.temperature = 1.0
    raw = oev._probs("state", pq)
    oev.temperature = 0.5
    sharp = oev._probs("state", pq)

    assert raw.index(max(raw)) == sharp.index(max(sharp))  # argmax unchanged
    assert max(sharp) > max(raw)  # <1 sharpens


def test_temperature_above_one_flattens():
    oev = _make_oev()
    q = {"type": "choice", "options": ["a", "b", "c"], "answer": "a"}
    pq = oev._question("q", q)

    oev.temperature = 1.0
    raw = oev._probs("state", pq)
    oev.temperature = 2.0
    flat = oev._probs("state", pq)

    assert max(flat) < max(raw)  # >1 flattens
    assert abs(sum(flat) - 1.0) < 1e-6  # still a distribution
