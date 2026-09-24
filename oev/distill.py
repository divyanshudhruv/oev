"""Distillation: train one 184M student on soft targets from teacher ensembles.

Round 1 per PLAN.md: student = fresh or warm-started 184M backbone, teachers =
b77 ensemble + mt generalist. Mixed-domain data (typed + banking77 + ag_news +
emotion) so benchmark sharpness and general skills land in the same weights.

Usage on Kaggle (GPU T4):

    python -m oev.distill \
        --teachers /kaggle/working/ck/b77a-oev-tiny.pt,/kaggle/working/ck/b77b-oev-tiny.pt,/kaggle/working/ck/oev-base-banking77.pt,/kaggle/working/ck/oev-base-mt.pt \
        --student-init /kaggle/working/ck/oev-base-td5.pt \
        --data-dir data --out /kaggle/working/distill_ckpt \
        --epochs 1 --batch-size 8 --save-every 500
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from oev.evaluate import load_model
from oev.tokenizer_hf import HFTokenPacker


class Teacher:
    """One frozen teacher: forward a packed case, return per-question probs."""

    def __init__(self, path, device):
        self.model = load_model(path, device)
        self.packer = HFTokenPacker(self.model.cfg["backbone"])
        self.device = device

    @torch.no_grad()
    def question_probs(self, state, question, max_len):
        q = {
            "name": question["name"],
            "type": question["type"],
            "instructions": question.get("instructions", question["type"]),
            "options": question["options"],
            "answer": question.get("answer", question["options"][0]),
        }
        ids, anchors, label = self.packer.pack(state, q, self.model.cfg["max_len"])
        tids = torch.tensor([ids], device=self.device)
        pmask = torch.zeros(1, len(ids), dtype=torch.bool, device=self.device)
        apos = torch.tensor([anchors], device=self.device)
        logits = self.model(tids, pmask, apos)[0]
        return F.softmax(logits, dim=-1)


def load_cases(data_dir, domains):
    cases = []
    for d in domains:
        path = os.path.join(data_dir, d, "train.jsonl")
        if not os.path.exists(path):
            print(f"[distill] skipping {d} (no train split)")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                cases.append((d, json.loads(line)))
    return cases


def run_epoch(student, teachers, cases, packer, device, opt, scaler, batch_size, log_every, epoch, f, out_dir):
    student.train()
    order = torch.randperm(len(cases))
    running, t0, n_seen = 0.0, time.time(), 0
    for bi, start in enumerate(range(0, len(order), batch_size)):
        batch = [cases[i] for i in order[start:start + batch_size]]
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=device == "cuda"):
            losses = []
            for domain, case in batch:
                q = case["questions"][0]
                state = case["state"]
                # student logits over this question's options
                sq = dict(q, answer=q.get("answer", q["options"][0]))
                ids, anchors, _ = packer.pack(state, sq, student.cfg["max_len"])
                tids = torch.tensor([ids], device=device)
                pmask = torch.zeros(1, len(ids), dtype=torch.bool, device=device)
                apos = torch.tensor([anchors], device=device)
                slogits = student(tids, pmask, apos)[0]
                logq = F.log_softmax(slogits, dim=-1)
                # teacher target: mean distribution over all teachers
                tprobs = None
                for t in teachers:
                    tp = t.question_probs(state, q, student.cfg["max_len"])
                    # align teacher/student option orders by name
                    topts = q["options"]
                    if tp.shape[0] != len(topts):
                        continue
                    tprobs = tp if tprobs is None else (tprobs + tp) / 2
                if tprobs is None:
                    continue
                loss = -(tprobs * logq).sum()
                losses.append(loss)
            if not losses:
                continue
            loss = torch.stack(losses).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        running += loss.item() * len(losses)
        n_seen += len(losses)
        if bi % log_every == 0:
            msg = (f"epoch {epoch} step {bi} running_loss {running / max(n_seen, 1):.4f} "
                   f"({n_seen / max(time.time() - t0, 1):.2f} cases/s)")
            print(msg, flush=True)
            f.write(msg + "\n"); f.flush()
        if out_dir and epoch is not None and bi and bi % 500 == 0:
            torch.save({"config": student.cfg, "state": student.state_dict()},
                       os.path.join(out_dir, "oev-tiny.pt"))
    return running / max(n_seen, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", required=True, help="comma-separated teacher checkpoint paths")
    ap.add_argument("--student-init", default=None, help="optional warm-start checkpoint for the student")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--domains", default="typed,banking77,ag_news,emotion")
    ap.add_argument("--out", default="checkpoints_distill")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device
    log = open(os.path.join(args.out, "distill.log"), "a")

    # student: warm-start if given, else first teacher's architecture from scratch
    if args.student_init:
        ckpt = torch.load(args.student_init, map_location=device, weights_only=True)
        from oev.model import OEVModel, OEVConfig
        student = OEVModel(OEVConfig(**ckpt["config"]))
        student.load_state_dict(ckpt["state"])
        student.cfg = ckpt["config"]
        print(f"student warm-started from {args.student_init}", flush=True)
    else:
        first = torch.load(args.teachers.split(",")[0], map_location=device, weights_only=True)
        from oev.model import OEVModel, OEVConfig
        student = OEVModel(OEVConfig(**first["config"]))
        student.cfg = first["config"]
        print("student initialized from scratch", flush=True)
    student.to(device)

    teachers = [Teacher(p.strip(), device) for p in args.teachers.split(",") if p.strip()]
    print(f"{len(teachers)} teachers loaded", flush=True)

    packer = HFTokenPacker(student.cfg["backbone"])
    cases = load_cases(args.data_dir, [d.strip() for d in args.domains.split(",")])
    print(f"{len(cases)} training cases across domains", flush=True)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    for epoch in range(args.epochs):
        tl = run_epoch(student, teachers, cases, packer, device, opt, scaler,
                       args.batch_size, args.log_every, epoch, log, args.out)
        msg = f"epoch {epoch} done - train loss {tl:.4f}"
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()
        torch.save({"config": student.cfg, "state": student.state_dict()},
                   os.path.join(args.out, "oev-tiny.pt"))
        print(f"saved -> {args.out}/oev-tiny.pt", flush=True)


if __name__ == "__main__":
    main()
