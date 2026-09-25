import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from oev.dataset import OEVDataset, collate
from oev.model import HFBackboneOEV, OEVConfig, OEVModel
from oev.tokenizer_hf import HFTokenPacker


def ece(confidences, corrects, bins=15):
    confidences = np.asarray(confidences)
    corrects = np.asarray(corrects)
    idx = np.minimum((confidences * bins).astype(int), bins - 1)
    total = len(confidences)
    value = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() > 0:
            value += (m.sum() / total) * abs(corrects[m].mean() - confidences[m].mean())
    return float(value)


def majority_baseline(answers):
    counts = Counter(answers)
    return counts.most_common(1)[0][1] / len(answers)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if "backbone" in ckpt["config"]:
        model = HFBackboneOEV(ckpt["config"]["backbone"])
        model.load_state_dict(ckpt["state"])
        model.eval().to(device)
        model.cfg = ckpt["config"]
        return model
    model = OEVModel(OEVConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state"])
    model.eval().to(device)
    return model


def collect(model, loader, device):
    correct = {"choice": [], "noul": [], "score": []}
    confs, corr = [], []
    with torch.no_grad():
        for batch in loader:
            logits = (
                model(batch["ids"].to(device), batch["pad_mask"].to(device), batch["anchor_pos"].to(device))
                + batch["logits_mask"].to(device)
            )
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            for i in range(len(batch["labels"])):
                t = batch["types"][i]
                ok = int(pred[i].item() == batch["labels"][i].item())
                correct[t].append(ok)
                confs.append(conf[i].item())
                corr.append(ok)
    return correct, confs, corr


def evaluate(checkpoint_path, data_dir="data", split="test", batch_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint_path, device)
    if isinstance(model, HFBackboneOEV):
        packer = HFTokenPacker(model.cfg["backbone"])
        ds = OEVDataset(f"{data_dir}/{split}.jsonl", model.cfg["max_len"], packer=packer)
    else:
        ds = OEVDataset(f"{data_dir}/{split}.jsonl", model.cfg.max_len)
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate)
    correct, confs, corr = collect(model, dl, device)
    answers = {}
    with open(f"{data_dir}/{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for q in row["questions"]:
                answers.setdefault(q["type"], []).append(q["answer"])
    results = {}
    for t in ("choice", "noul", "score"):
        if correct[t]:
            results[f"accuracy_{t}"] = sum(correct[t]) / len(correct[t])
            results[f"baseline_{t}"] = majority_baseline(answers[t])
    results["ece"] = ece(confs, corr)
    results["params"] = sum(p.numel() for p in model.parameters())
    results["disk_mb"] = os.path.getsize(checkpoint_path) / 1e6
    ids = torch.tensor([[1, 5, 2, 3, 7, 3, 8]], device=device)
    pad = torch.zeros(1, 7, dtype=torch.bool, device=device)
    apos = torch.tensor([[3, 5]], device=device)
    with torch.no_grad():
        model(ids, pad, apos)
        t0 = time.perf_counter()
        for _ in range(100):
            model(ids, pad, apos)
        results["latency_ms_1q"] = (time.perf_counter() - t0) * 10
        big_ids = ids.repeat(64, 1)
        big_pad = pad.repeat(64, 1)
        big_apos = apos.repeat(64, 1)
        model(big_ids, big_pad, big_apos)
        t0 = time.perf_counter()
        for _ in range(20):
            model(big_ids, big_pad, big_apos)
        results["latency_ms_per_q_batched"] = (time.perf_counter() - t0) * 1000 / (20 * 64)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/oev-tiny.pt")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()
    r = evaluate(args.checkpoint, data_dir=args.data_dir)
    lines = ["| metric | value |", "|---|---|"]
    for k, v in r.items():
        lines.append(f"| {k} | {v:.4f} |")
    out = Path("benchmarks")
    out.mkdir(exist_ok=True)
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
