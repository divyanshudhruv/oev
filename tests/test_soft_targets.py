import torch
from oev.dataset import OEVDataset, collate
from oev.data_gen import generate, write_splits


def make_typed_file(tmp_path):
    import json

    rows = [
        {
            "id": "t-0",
            "domain": "wf",
            "state": "s",
            "questions": [
                {"name": "d", "type": "choice", "options": ["a", "b", "c", "d"], "answer": "a", "target": [0.7, 0.1, 0.1, 0.1]},
                {"name": "u", "type": "noul", "options": ["no", "yes"], "answer": "no", "target": [0.6, 0.4]},
            ],
        }
    ]
    p = tmp_path / "typed"
    p.mkdir()
    with open(p / "train.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def test_collate_soft_targets(tmp_path):
    p = make_typed_file(tmp_path)
    ds = OEVDataset(str(p / "train.jsonl"), 128)
    assert ds[0]["target"] == [0.7, 0.1, 0.1, 0.1]
    b = collate([ds[0], ds[1]])
    assert b["has_target"].tolist() == [True, True]
    assert torch.allclose(b["targets"][0, :4], torch.tensor([0.7, 0.1, 0.1, 0.1]))
    assert torch.allclose(b["targets"][1, :2], torch.tensor([0.6, 0.4]))


def test_collate_hard_target_fallback(tmp_path):
    import json

    p = tmp_path / "hard"
    p.mkdir()
    rows = [{"id": "h-0", "domain": "x", "state": "s", "questions": [{"name": "d", "type": "choice", "options": ["a", "b"], "answer": "b"}]}]
    with open(p / "train.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ds = OEVDataset(str(p / "train.jsonl"), 64)
    b = collate([ds[0]])
    assert b["has_target"].tolist() == [False]
    assert b["targets"].sum() == 0


def test_mixed_batch_shapes():
    splits = generate(30, seed=5)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        write_splits(splits, d)
        ds = OEVDataset(f"{d}/train.jsonl", 128)
        b = collate([ds[i] for i in range(4)])
        assert b["has_target"].tolist() == [False] * 4
