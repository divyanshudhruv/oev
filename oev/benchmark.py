import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from oev.dataset import OEVDataset, collate
from oev.model import HFBackboneOEV
from oev.evaluate import load_model, ece
from oev.calibrate import fit_temperature_from_logits
from oev.tokenizer_hf import HFTokenPacker


def evaluate_benchmark(checkpoint, data_dir, calibrate=True, batch_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint, device)
    if isinstance(model, HFBackboneOEV):
        packer = HFTokenPacker(model.cfg["backbone"])
        max_len = model.cfg["max_len"]
    else:
        packer = None
        max_len = model.cfg.max_len
    dl = DataLoader(OEVDataset(f"{data_dir}/test.jsonl", max_len, packer=packer), batch_size=batch_size, collate_fn=collate)
    confs, corr = [], []
    val_logits, val_labels = [], []
    if calibrate:
        vdl = DataLoader(OEVDataset(f"{data_dir}/valid.jsonl", max_len, packer=packer), batch_size=batch_size, collate_fn=collate)
        with torch.no_grad():
            for batch in vdl:
                logits = model(batch["ids"].to(device), batch["pad_mask"].to(device), batch["anchor_pos"].to(device)) + batch["logits_mask"].to(device)
                keep = batch["anchor_valid"].sum(dim=1)
                for i in range(logits.size(0)):
                    row = logits[i][: keep[i]]
                    if row.numel() >= 2:
                        val_logits.append(row)
                        val_labels.append(batch["labels"][i])
        temperature = fit_temperature_from_logits(torch.stack(val_logits), torch.stack(val_labels))
    else:
        temperature = 1.0
    with torch.no_grad():
        for batch in dl:
            logits = model(batch["ids"].to(device), batch["pad_mask"].to(device), batch["anchor_pos"].to(device)) + batch["logits_mask"].to(device)
            probs = F.softmax(logits / temperature, dim=-1)
            conf, pred = probs.max(dim=-1)
            for i in range(len(batch["labels"])):
                confs.append(conf[i].item())
                corr.append(int(pred[i].item() == batch["labels"][i].item()))
    return {
        "accuracy": sum(corr) / len(corr),
        "n": len(corr),
        "ece": ece(confs, corr),
        "temperature": temperature,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--no-calibrate", action="store_true")
    args = p.parse_args()
    r = evaluate_benchmark(args.checkpoint, args.data_dir, calibrate=not args.no_calibrate)
    for k, v in r.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
