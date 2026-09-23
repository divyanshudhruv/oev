import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from oev.dataset import OEVDataset, collate
from oev.evaluate import load_model
from oev.tokenizer_hf import HFTokenPacker


def brier(probs, target):
    """Mean Brier score between predicted and reference distributions (lower is better)."""
    return ((probs - target) ** 2).sum(-1).mean()


def rlcd_step(model, batch, device, opt, scaler, alpha=0.5):
    """One RLCD update.

    Phase 1 (imitation anchor): soft cross-entropy against the teacher's
    distribution — keeps the model from drifting off the teacher entirely.
    Phase 2 (Brier improvement): evaluate the Brier score of the model's own
    predicted distribution against the teacher's reference; treat (1 - Brier)
    as the reward and follow its gradient (REINFORCE-style, on the argmax
    path the model actually commits to).

    alpha blends the two losses.
    """
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=scaler is not None and device == "cuda"):
        logits = model(batch["ids"], batch["pad_mask"], batch["anchor_pos"]) + batch["logits_mask"]
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)  # guard fp16 overflow upstream
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = torch.nan_to_num(log_probs, neginf=-1e4)
        probs = log_probs.exp()

        # Phase 1: imitation anchor (soft CE on rows that carry targets)
        soft = -(batch["targets"] * log_probs).sum(-1)
        anchor = soft[batch["has_target"]].mean() if batch["has_target"].any() else soft.mean()

        # Phase 2: advantage = Brier reward centered by batch mean (baseline).
        # Without centering, the constant positive reward just amplifies the
        # current argmax — the collapse we saw at 0.41 accuracy.
        tgt = batch["targets"]
        b = brier(probs, tgt)
        advantage = ((1.0 - b) - (1.0 - b).detach().mean()).detach()
        # score-function gradient: raise probability of the model's own current
        # answer proportional to how well its distribution scored
        picked = probs.argmax(-1)
        score = -log_probs.gather(1, picked.unsqueeze(1)).squeeze(1)
        rl = -(advantage * score).mean()

        loss = alpha * anchor + (1 - alpha) * rl
    opt.zero_grad()
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return loss.item(), b.item()


def rlcd(checkpoint, data_dir="data/typed", epochs=2, batch_size=8, lr=5e-5, max_len=None, out="checkpoints_rlcd", alpha=0.5, seed=0):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint, device)
    max_len = max_len or model.cfg["max_len"]
    backbone = model.cfg["backbone"]
    packer = HFTokenPacker(backbone)

    ds = OEVDataset(f"{data_dir}/train.jsonl", max_len, packer=packer)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    steps = len(dl)
    for epoch in range(epochs):
        tot = bsum = n = 0
        for i, batch in enumerate(dl):
            l, b = rlcd_step(model, batch, device, opt, scaler, alpha=alpha)
            tot += l
            bsum += b
            n += 1
            if (i + 1) % 50 == 0:
                print(f"epoch {epoch} step {i + 1}/{steps} loss {tot / n:.4f} brier {bsum / n:.4f}", flush=True)
        torch.save({"config": model.cfg, "state": model.state_dict()}, out / "oev-tiny.pt")
        print(f"epoch {epoch} avg_loss {tot / n:.4f} avg_brier {bsum / n:.4f} -> saved", flush=True)
    print("rlcd done", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="data/typed")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--out", default="checkpoints_rlcd")
    p.add_argument("--alpha", type=float, default=0.5, help="imitation anchor weight (1 = pure imitation)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rlcd(checkpoint=args.checkpoint, data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_len=args.max_len, out=args.out, alpha=args.alpha, seed=args.seed)
