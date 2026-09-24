"""Weight-average (soup) multiple warm-started checkpoints into one.

Works because the inputs share a loss basin (warm-started from the same
parent). Run on checkpoints that fine-tuned from a common init:

    python -m oev.soup --checkpoints a.pt,b.pt,c.pt --out soup.pt
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True, help="comma-separated .pt paths sharing an architecture")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = [p.strip() for p in args.checkpoints.split(",") if p.strip()]
    assert len(paths) >= 2, "soup needs at least two checkpoints"

    states = []
    config = None
    for p in paths:
        ckpt = torch.load(p, map_location="cpu", weights_only=True)
        if config is None:
            config = ckpt.get("config")
        states.append(ckpt["state"])

    keys = states[0].keys()
    for s in states[1:]:
        assert s.keys() == keys, "checkpoint state dicts disagree; soup only works within one training lineage"

    soup = {}
    for k in keys:
        acc = states[0][k].float()
        for s in states[1:]:
            acc = acc + s[k].float()
        soup[k] = (acc / len(states)).to(states[0][k].dtype)

    torch.save({"config": config, "state": soup}, args.out)
    print(f"soup of {len(paths)} checkpoints -> {args.out} "
          f"({sum(v.numel() for v in soup.values())} params averaged)")


if __name__ == "__main__":
    main()
