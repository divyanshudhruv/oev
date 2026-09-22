import json
import pytest

transformers = pytest.importorskip("transformers")
import tokenizers
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, AutoConfig, AutoModel
import torch
from oev.tokenizer_hf import HFTokenPacker
from oev.model import HFBackboneOEV
from oev.train import train
from oev.infer import OEV


@pytest.fixture(scope="module")
def backbone_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("bb_env")
    bb_dir = root / "bb"
    bb_dir.mkdir()
    vocab = {"<pad>": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[ANCHOR]": 4, "h": 5, "e": 6, "l": 7, "o": 8, "w": 9, "r": 10, "d": 11, " ": 12, "a": 13, "b": 14, "c": 15, "1": 16, "2": 17, "3": 18, ":": 19}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]")
    tok.save_pretrained(str(bb_dir))
    cfg = AutoConfig.for_model("bert", hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, vocab_size=len(vocab) + 10)
    AutoModel.from_config(cfg).save_pretrained(str(bb_dir))
    data = root / "data"
    data.mkdir()
    rows = [
        {"id": f"r-{i}", "domain": "toy", "state": "hello world abc" * (1 + i % 2), "questions": [{"name": "pick", "type": "choice", "options": ["a", "b"], "answer": "a" if i % 2 == 0 else "b"}]}
        for i in range(12)
    ]
    for split, rows_in in (("train", rows[:8]), ("valid", rows[8:10]), ("test", rows[10:])):
        with open(data / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows_in:
                f.write(json.dumps(r) + "\n")
    return str(bb_dir), str(data), str(root / "ckpt")


def test_backbone_roundtrip(backbone_env, monkeypatch):
    bb_dir, data_dir, out_dir = backbone_env

    real_tok = PreTrainedTokenizerFast.from_pretrained(bb_dir)
    real_model = AutoModel.from_pretrained(bb_dir)

    def fake_autoTokenizer(name, **kwargs):
        return real_tok

    def fake_autoModel(name, **kwargs):
        return AutoModel.from_pretrained(bb_dir, **{}) if False else real_model

    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_autoTokenizer)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", fake_autoModel)

    train(preset="tiny", epochs=1, batch_size=4, data_dir=data_dir, out=out_dir, backbone=bb_dir)
    ckpt_file = out_dir + "/oev-tiny.pt"
    agent = OEV(ckpt_file, device="cpu")
    out = agent.decide("hello world", {"pick": {"type": "choice", "options": ["a", "b"]}})
    assert "pick" in out
    assert abs(sum(out["pick"]["probabilities"].values()) - 1.0) < 1e-4
