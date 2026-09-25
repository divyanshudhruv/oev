import argparse
import json

import torch

from oev.evaluate import load_model
from oev.tokenizer_hf import HFTokenPacker


def ensemble_accuracy(ckpts, data_dir, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    models = [load_model(c, device) for c in ckpts]
    packers = [HFTokenPacker(m.cfg["backbone"]) for m in models]
    with open(f"{data_dir}/test.jsonl", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]

    correct = total = 0
    with torch.no_grad():
        for r in rows:
            for q in r["questions"]:
                pq = {
                    "name": q["name"],
                    "type": q["type"],
                    "instructions": q.get("instructions", q["type"]),
                    "options": q["options"],
                    "answer": q["answer"],
                }
                probs_sum = None
                label = None
                for m, p in zip(models, packers):
                    ids, anchors, label = p.pack(r["state"], pq, m.cfg["max_len"])
                    tids = torch.tensor([ids], device=device)
                    pmask = torch.zeros(1, len(ids), dtype=torch.bool, device=device)
                    apos = torch.tensor([anchors], device=device)
                    pr = torch.softmax(m(tids, pmask, apos)[0].float(), dim=-1)
                    probs_sum = pr if probs_sum is None else probs_sum + pr
                total += 1
                correct += int(probs_sum.argmax().item() == label)
    return correct / total, total


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", required=True, help="comma-separated checkpoint paths")
    p.add_argument("--data-dir", default="data/typed")
    args = p.parse_args()
    ckpts = [c.strip() for c in args.ckpts.split(",") if c.strip()]
    acc, n = ensemble_accuracy(ckpts, args.data_dir)
    print(f"ensemble accuracy: {acc:.4f} (n={n})")
