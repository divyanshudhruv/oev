import pytest

transformers = pytest.importorskip("transformers")
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from oev.tokenizer_hf import HFTokenPacker


@pytest.fixture(scope="module")
def tiny_packer(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    vocab = {"<pad>": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[ANCHOR]": 4, "h": 5, "e": 6, "l": 7, "o": 8, "w": 9, "r": 10, "d": 11, " ": 12, "a": 13, "b": 14, "c": 15, "1": 16, "2": 17, "3": 18, ":": 19}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]")
    tok.save_pretrained(str(d))
    return HFTokenPacker(str(d))


def test_pack_anchor_positions(tiny_packer):
    q = {"name": "pick", "type": "choice", "instructions": "choose", "options": ["a", "b"], "answer": "b"}
    ids, anchors, label = tiny_packer.pack("hello", q, max_len=128)
    assert len(anchors) == 2
    assert label == 1
    assert all(ids[p] == tiny_packer.anchor_id for p in anchors)
