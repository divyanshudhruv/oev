import json

import torch
from torch.utils.data import Dataset

from oev import tokenizer as tk


def pack(state, question, max_len=512):
    max_len = max(1, int(max_len))
    options = question["options"]
    q_ids = tk.encode(question["name"] + ": " + question.get("instructions", question["type"]))
    opt_ids = [[tk.ANCHOR_ID] + tk.encode(" " + o) for o in options]
    fixed = 2 + len(q_ids) + sum(len(o) for o in opt_ids)
    budget = max(1, max_len - fixed)
    s_ids = ([tk.CLS_ID] + tk.encode(state) + [tk.SEP_ID])[:budget]
    ids = s_ids + q_ids
    anchor_pos = []
    for o in opt_ids:
        anchor_pos.append(len(ids))
        ids = ids + o
    ids = ids[:max_len]
    anchor_pos = [min(anchor, len(ids) - 1) for anchor in anchor_pos]
    label = options.index(question["answer"])
    return ids, anchor_pos, label


class OEVDataset(Dataset):
    def __init__(self, path, max_len=512, packer=None):
        self.max_len = max_len
        self.packer = packer
        self.items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                for q in row["questions"]:
                    self.items.append((row["state"], q))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        state, q = self.items[i]
        if self.packer is not None:
            ids, anchors, label = self.packer.pack(state, q, self.max_len)
        else:
            ids, anchors, label = pack(state, q, self.max_len)
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "anchors": torch.tensor(anchors, dtype=torch.long),
            "label": label,
            "type": q["type"],
            "n": len(anchors),
            "target": q.get("target"),
        }


def collate(batch):
    B = len(batch)
    L = max(len(b["ids"]) for b in batch)
    A = max(b["n"] for b in batch)
    ids = torch.full((B, L), tk.PAD_ID, dtype=torch.long)
    pad_mask = torch.ones(B, L, dtype=torch.bool)
    anchor_pos = torch.zeros(B, A, dtype=torch.long)
    anchor_valid = torch.zeros(B, A, dtype=torch.bool)
    logits_mask = torch.full((B, A), float("-inf"))
    labels = torch.zeros(B, dtype=torch.long)
    types = []
    targets = torch.zeros(B, A, dtype=torch.float32)
    has_target = torch.zeros(B, dtype=torch.bool)
    for i, b in enumerate(batch):
        n, l = b["n"], len(b["ids"])
        ids[i, :l] = b["ids"]
        pad_mask[i, :l] = False
        anchor_pos[i, :n] = b["anchors"]
        anchor_valid[i, :n] = True
        logits_mask[i, :n] = 0.0
        labels[i] = b["label"]
        types.append(b["type"])
        if b.get("target") is not None:
            t = b["target"]
            targets[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
            has_target[i] = True
    return {
        "ids": ids,
        "pad_mask": pad_mask,
        "anchor_pos": anchor_pos,
        "anchor_valid": anchor_valid,
        "logits_mask": logits_mask,
        "labels": labels,
        "types": types,
        "targets": targets,
        "has_target": has_target,
    }
