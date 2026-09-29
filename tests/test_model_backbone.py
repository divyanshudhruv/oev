import pytest

transformers = pytest.importorskip("transformers")
import torch
from transformers import AutoConfig, AutoModel

from oev.model import HFBackboneOEV


@pytest.fixture(scope="module")
def tiny_backbone(tmp_path_factory):
    d = tmp_path_factory.mktemp("bb")
    cfg = AutoConfig.for_model("bert", hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64, vocab_size=50)
    AutoModel.from_config(cfg).save_pretrained(str(d))
    return str(d)


def test_forward_shape_and_padding_invariance(tiny_backbone):

    m = HFBackboneOEV(tiny_backbone)
    m.eval()
    ids = torch.randint(5, 50, (2, 16))
    pad = torch.zeros(2, 16, dtype=torch.bool)
    pad[1, 12:] = True
    apos = torch.tensor([[2, 6, 10], [3, 7, 11]])
    full = m(ids, pad, apos)
    assert full.shape == (2, 3)
    pad2 = pad.clone()
    pad2[1, 8:] = True
    trimmed = m(ids, pad2, apos)
    assert torch.allclose(full[0], trimmed[0], atol=1e-4)
