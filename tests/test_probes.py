import pytest
import torch

from oev.model import OEVConfig, OEVModel, PRESETS
from oev.probes import _max_len, forgery, order


@pytest.fixture(scope="module")
def tiny_model():
    cfg = OEVConfig(**PRESETS["tiny"], max_len=512)
    model = OEVModel(cfg)
    model.eval()
    return model


class CharPacker:
    # probes only need pack() and the anchor id for the char pipeline
    import oev.tokenizer as _tk

    anchor_id = _tk.ANCHOR_ID

    class tok:
        pass

    def pack(self, state, question, max_len):
        from oev.dataset import pack

        return pack(state, question, max_len)


CharPacker.tok.convert_ids_to_tokens = staticmethod(lambda ids: ["<anchor>"])


def test_max_len_handles_both_config_shapes(tiny_model):
    assert _max_len(tiny_model) == 512


def test_max_len_dict_config():
    class DictModel:
        cfg = {"max_len": 768}

    assert _max_len(DictModel()) == 768


def test_forgery_anchors_match_options(tiny_model):
    # every adversarial option list must yield exactly one scored anchor
    # per option, regardless of anchor tokens or delimiter text inside
    assert forgery(tiny_model, CharPacker(), "cpu") is True


def test_order_returns_a_rate(tiny_model):
    rate = order(tiny_model, CharPacker(), "cpu", rotations=3)
    assert 0.0 <= rate <= 1.0


def test_anchor_overflow_is_rejected(tiny_model):
    # a probe question too long for max_len must raise, not silently score
    # out-of-bounds anchors
    from oev.probes import _predict

    big_q = {
        "name": "q",
        "type": "choice",
        "instructions": "x" * 2000,
        "options": [f"option {i}" for i in range(8)],
        "answer": "option 0",
    }
    with pytest.raises(ValueError):
        _predict(tiny_model, CharPacker(), "state", big_q, "cpu")
