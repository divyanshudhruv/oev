import json

import torch

from oev.dataset import OEVDataset, collate, pack


def q():
    return {"name": "dept", "type": "choice", "instructions": "pick one", "options": ["a", "b", "c"], "answer": "b"}


def test_pack_anchor_count_and_label():
    ids, anchors, label = pack("the server is down", q(), 512)
    assert len(anchors) == 3
    assert label == 1
    assert ids[0] == 1 and 2 in ids[:20]


def test_pack_truncates_state_not_options():
    long_state = "x" * 5000
    ids, anchors, _label = pack(long_state, q(), 512)
    assert len(ids) <= 512 and len(anchors) == 3


def test_pack_clamps_anchors_after_short_clip():
    long_question = {
        "name": "department",
        "type": "choice",
        "instructions": "choose " * 100,
        "options": ["a" * 100, "b" * 100, "c" * 100],
        "answer": "b" * 100,
    }

    ids, anchors, _ = pack("state", long_question, 8)

    assert len(ids) == 8
    assert all(0 <= anchor < len(ids) for anchor in anchors)


def test_collate_shapes_and_masks():
    b1 = {"ids": torch.tensor([1, 5, 6, 2, 3, 7]), "anchors": torch.tensor([4, 5]), "label": 0, "type": "choice", "n": 2}
    b2 = {"ids": torch.tensor([1, 8, 2, 3, 9, 3, 10, 3, 11]), "anchors": torch.tensor([3, 5, 7]), "label": 2, "type": "noul", "n": 3}
    out = collate([b1, b2])
    assert out["ids"].shape == (2, 9)
    assert out["anchor_pos"].shape == (2, 3)
    assert out["labels"].tolist() == [0, 2]
    assert out["logits_mask"][0, 2].item() == float("-inf")
    assert out["logits_mask"][1, 2].item() == 0.0
    assert out["pad_mask"][0, 6:].all() and not out["pad_mask"][0, 5]
    assert out["types"] == ["choice", "noul"]


def test_dataset_expands_questions(tmp_path):
    row = {"id": "s-0", "domain": "review", "state": "it was great", "questions": [q(), {"name": "r", "type": "score", "options": ["1", "2", "3"], "answer": "2"}]}
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    ds = OEVDataset(str(p))
    assert len(ds) == 2
    item = ds[1]
    assert item["n"] == 3 and item["type"] == "score"
