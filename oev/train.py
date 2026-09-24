import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from oev.dataset import OEVDataset, collate
from oev.model import OEVModel, OEVConfig, PRESETS, HFBackboneOEV
from oev.tokenizer_hf import HFTokenPacker


def run_epoch(model, loader, device, opt=None, log_every=0, epoch=0, scaler=None):
    model.train(opt is not None)
    no_grad = opt is None
    total = 0.0
    n = 0
    steps = 0
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        cm = torch.no_grad() if no_grad else torch.enable_grad()
        with cm, torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            logits = model(batch["ids"], batch["pad_mask"], batch["anchor_pos"]) + batch["logits_mask"]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            log_probs = torch.nan_to_num(log_probs, neginf=-1e4)  # 0 * -inf = NaN on masked options
            hard = F.nll_loss(log_probs, batch["labels"], reduction="none")
            soft = -(batch["targets"] * log_probs).sum(-1)
            per = torch.where(batch["has_target"], soft, hard)
            loss = per.sum()
        if opt is not None:
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
        total += loss.item()
        n += batch["labels"].numel()
        steps += 1
        if log_every and steps % log_every == 0:
            print(f"epoch {epoch} step {steps} running_loss {total / n:.4f}", flush=True)
    return total / n


def make_loaders(data_dirs, max_len, batch_size, packer=None):
    tr, va = [], []
    for d in data_dirs:
        tr.append(OEVDataset(f"{d}/train.jsonl", max_len, packer=packer))
        va.append(OEVDataset(f"{d}/valid.jsonl", max_len, packer=packer))
    train_dl = DataLoader(ConcatDataset(tr), batch_size=batch_size, shuffle=True, collate_fn=collate)
    valid_dl = DataLoader(ConcatDataset(va), batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_dl, valid_dl


def train(preset="tiny", epochs=5, batch_size=64, lr=3e-4, seed=0, max_len=512, data_dir="data", out="checkpoints", backbone=None, init=None):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dirs = [d.strip() for d in data_dir.split(",") if d.strip()]
    if backbone:
        model = HFBackboneOEV(backbone).to(device)
        if init:
            ckpt = torch.load(init, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["state"])
            print(f"initialized from {init}")
        model.backbone.float()
        packer = HFTokenPacker(backbone)
        train_dl, valid_dl = make_loaders(data_dirs, max_len, batch_size, packer=packer)
        groups = [
            {"params": list(model.backbone.parameters()), "lr": 2e-5},
            {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone")], "lr": 1e-3},
        ]
        opt = torch.optim.AdamW(groups, weight_decay=0.01)
        scaler = torch.amp.GradScaler(enabled=device == "cuda")
    else:
        cfg = OEVConfig(**PRESETS[preset], max_len=max_len)
        model = OEVModel(cfg).to(device)
        train_dl, valid_dl = make_loaders(data_dirs, cfg.max_len, batch_size)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scaler = None
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(epochs):
        tl = run_epoch(model, train_dl, device, opt, log_every=50, epoch=epoch, scaler=scaler)
        vl = run_epoch(model, valid_dl, device)
        sched.step()
        print(f"epoch {epoch} train_loss {tl:.4f} valid_loss {vl:.4f}", flush=True)
        if vl < best:
            best = vl
            if backbone:
                torch.save({"config": {"backbone": backbone, "max_len": max_len}, "state": model.state_dict()}, out / f"oev-{preset}.pt")
            else:
                torch.save({"config": {**PRESETS[preset], "max_len": max_len}, "state": model.state_dict()}, out / f"oev-{preset}.pt")
        if device == "cuda":
            torch.cuda.empty_cache()
    print("best_valid_loss", f"{best:.4f}", flush=True)
    return best


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="tiny", choices=["tiny", "base"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--data-dir", default="data", help="comma-separated list of dataset dirs")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--backbone", default=None)
    p.add_argument("--init", default=None, help="checkpoint to initialize weights from (staged fine-tuning)")
    args = p.parse_args()
    train(preset=args.preset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_len=args.max_len, data_dir=args.data_dir, out=args.out, backbone=args.backbone, init=args.init)
