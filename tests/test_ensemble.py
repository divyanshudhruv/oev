import json

import torch

from oev import ensemble


class _Model:
    cfg = {"backbone": "fake", "max_len": 8}

    def __call__(self, ids, pad_mask, anchor_pos):
        return torch.tensor([[0.0, 1.0]])


class _Packer:
    def pack(self, state, question, max_len):
        return [1, 2, 3], [1, 2], question["options"].index(question["answer"])


def test_ensemble_falls_back_to_cpu_before_loading(monkeypatch, tmp_path):
    calls = []
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.jsonl").write_text(
        json.dumps({
            "state": "state",
            "questions": [{
                "name": "choice",
                "type": "choice",
                "options": ["no", "yes"],
                "answer": "yes",
            }],
        }) + "\n",
        encoding="utf-8",
    )

    def fake_load(checkpoint, device):
        calls.append(device)
        return _Model()

    monkeypatch.setattr(ensemble, "load_model", fake_load)
    monkeypatch.setattr(ensemble, "HFTokenPacker", lambda name: _Packer())
    monkeypatch.setattr(ensemble.torch.cuda, "is_available", lambda: False)

    accuracy, total = ensemble.ensemble_accuracy(["checkpoint"], str(data_dir), device="cuda")

    assert calls == ["cpu"]
    assert accuracy == 1.0
    assert total == 1
