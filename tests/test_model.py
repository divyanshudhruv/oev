import torch

from oev.model import PRESETS, OEVConfig, OEVModel


def small():
    return OEVConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64)


def batch(n2_padded=20):
    ids = torch.randint(4, 99, (2, 20))
    ids[:, 0] = 1
    pad = torch.zeros(2, 20, dtype=torch.bool)
    pad[1, n2_padded:] = True
    apos = torch.tensor([[2, 6, 10], [3, 7, 11]])
    return ids, pad, apos


def test_forward_shape():
    m = OEVModel(small())
    ids, pad, apos = batch()
    assert m(ids, pad, apos).shape == (2, 3)


def test_padding_does_not_change_other_row():
    m = OEVModel(small())
    m.eval()
    ids, pad, apos = batch()
    full = m(ids, pad, apos)
    pad2 = pad.clone()
    pad2[1, 12:] = True
    trimmed = m(ids, pad2, apos)
    assert torch.allclose(full[0], trimmed[0], atol=1e-5)


def test_param_counts():
    tiny = sum(p.numel() for p in OEVModel(OEVConfig(**PRESETS["tiny"])).parameters())
    base = sum(p.numel() for p in OEVModel(OEVConfig(**PRESETS["base"])).parameters())
    assert 100_000 < tiny < 600_000
    assert 5_000_000 < base < 20_000_000
