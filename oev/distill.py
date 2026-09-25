import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from oev.evaluate import load_model
from oev.tokenizer_hf import HFTokenPacker


class Teacher:
    # one frozen teacher: forward a packed case, return per-question probs

    def __init__(self, path, device):
        self.model = load_model(path, device)
        self.packer = HFTokenPacker(self.model.cfg["backbone"])
        self.device = device

    @torch.no_grad()
    def question_probs(self, state, question, max_len, tau):
        q = {
            "name": question["name"],
            "type": question["type"],
            "instructions": question.get("instructions", question["type"]),
            "options": question["options"],
            "answer": question.get("answer", question["options"][0]),
        }
        ids, anchors, _ = self.packer.pack(state, q, min(max_len, self.model.cfg["max_len"]))
        tids = torch.tensor([ids], device=self.device)
        pmask = torch.zeros(1, len(ids), dtype=torch.bool, device=self.device)
        apos = torch.tensor([anchors], device=self.device)
        logits = self.model(tids, pmask, apos)[0]
        return F.softmax(logits.float() / tau, dim=-1)


def rotate(q, rng, p):
    # order-invariance augmentation: rotate options + answer in lockstep
    n = len(q["options"])
    if n < 3 or rng.random() > p:
        return q
    k = rng.randrange(1, n)
    ops = q["options"]
    rot = ops[k:] + ops[:k]
    out = dict(q)
    out["options"] = rot
    ans = q.get("answer")
    if ans in ops:
        out["answer"] = rot[(ops.index(ans) - k) % n]
    return out


def load_cases(data_dir, domains, max_cases, seed, per_domain=0):
    """Load train cases; with per_domain>0, balance every domain to that count
    (big domains sampled down, small domains repeated -- rotation augmentation
    makes repeats non-identical)."""
    rng = random.Random(seed)
    pools = []
    for d in domains:
        path = os.path.join(data_dir, d, "train.jsonl")
        if not os.path.exists(path):
            print(f"[distill] skipping {d} (no train split)", flush=True)
            continue
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        if per_domain:
            if len(rows) >= per_domain:
                rows = rng.sample(rows, per_domain)
            else:
                rows = (rows * (per_domain // len(rows) + 1))[:per_domain]
            print(f"[distill] {d}: {len(rows)} cases (balanced)", flush=True)
        pools.append(rows)
    cases = [c for rows in pools for c in rows]
    rng.shuffle(cases)
    if max_cases and len(cases) > max_cases:
        cases = cases[:max_cases]
    return cases


def run_epoch(student, teachers, cases, packer, device, opt, scaler, args, epoch, f):
    student.train()
    rng = random.Random(epoch + 1)
    order = torch.randperm(len(cases))
    running, t0, n_seen = 0.0, time.time(), 0
    running_ce, running_kl, n_ce, n_kl = 0.0, 0.0, 0, 0
    for bi, start in enumerate(range(0, len(order), args.batch_size)):
        batch = [cases[i] for i in order[start:start + args.batch_size].tolist()]
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=device == "cuda"):
            losses, ce_parts, kl_parts = [], [], []
            for case in batch:
                q = rotate(case["questions"][0], rng, args.rotate)
                state = case["state"]
                sq = dict(q, answer=q.get("answer", q["options"][0]))
                ids, anchors, _ = packer.pack(state, sq, student.cfg["max_len"])
                tids = torch.tensor([ids], device=device)
                pmask = torch.zeros(1, len(ids), dtype=torch.bool, device=device)
                apos = torch.tensor([anchors], device=device)
                slogits = student(tids, pmask, apos)[0]
                logq = F.log_softmax(slogits.float() / args.tau, dim=-1)
                tprobs, n_scored = None, 0
                for t in teachers:
                    tp = t.question_probs(state, q, student.cfg['max_len'], args.tau)
                    if tp.shape[0] == len(q["options"]):
                        tprobs = tp if tprobs is None else (tprobs + tp)
                        n_scored += 1
                if tprobs is None:
                    continue
                tprobs = tprobs / n_scored  # mean over teachers that actually scored this option set
                kl = -(tprobs * logq).sum() * (args.tau ** 2)
                kl_parts.append(kl)
                ans = q.get("answer")
                if args.alpha and ans in q["options"]:
                    gold = q["options"].index(ans)
                    ce = F.cross_entropy(slogits.float().unsqueeze(0),
                                         torch.tensor([gold], device=device))
                    ce_parts.append(ce)
                    losses.append((1.0 - args.alpha) * kl + args.alpha * ce)
                else:
                    losses.append(kl)
            if not losses:
                continue
            loss = torch.stack(losses).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        running += loss.item() * len(losses)
        n_seen += len(losses)
        if kl_parts:
            running_kl += torch.stack(kl_parts).sum().item()
            n_kl += len(kl_parts)
        if ce_parts:
            running_ce += torch.stack(ce_parts).sum().item()
            n_ce += len(ce_parts)
        if bi % args.log_every == 0:
            msg = (f"epoch {epoch} step {bi} loss {running / max(n_seen, 1):.4f} "
                   f"(ce {running_ce / max(n_ce, 1):.3f} kl {running_kl / max(n_kl, 1):.3f}) "
                   f"({n_seen / max(time.time() - t0, 1):.2f} cases/s)")
            print(msg, flush=True)
            f.write(msg + "\n")
            f.flush()
        if args.save_every and bi and bi % args.save_every == 0:
            torch.save({"config": student.cfg, "state": student.state_dict()},
                       os.path.join(args.out, f"student-e{epoch}-s{bi}.pt"))
    return running / max(n_seen, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", required=True, help="comma-separated teacher checkpoint paths")
    ap.add_argument("--student-init", default=None, help="optional warm-start checkpoint for the student")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--domains", default="banking77,ag_news,emotion",
                    help="comma-separated domain dirs under --data-dir, each with train.jsonl")
    ap.add_argument("--out", default="checkpoints_distill")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--rotate", type=float, default=0.5, help="option-rotation augmentation probability")
    ap.add_argument("--tau", type=float, default=1.0, help="distillation temperature")
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="weight on gold cross-entropy; (1-alpha) goes to teacher KL. 0 = pure mimicry")
    ap.add_argument("--per-domain", type=int, default=0,
                    help="if >0, balance each domain to this many train cases")
    ap.add_argument("--max-cases", type=int, default=27000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device
    log = open(os.path.join(args.out, "distill.log"), "a")  # noqa: SIM115 - kept open for the whole run

    if args.student_init:
        student = load_model(args.student_init, device)
        print(f"student warm-started from {args.student_init}", flush=True)
    else:
        first = load_model(args.teachers.split(",")[0].strip(), device)
        from oev.model import HFBackboneOEV
        student = HFBackboneOEV(first.cfg["backbone"])
        student.cfg = first.cfg
        student.eval().to(device)
        print("student initialized from scratch", flush=True)

    teachers = [Teacher(p.strip(), device) for p in args.teachers.split(",") if p.strip()]
    print(f"{len(teachers)} teachers loaded", flush=True)

    packer = HFTokenPacker(student.cfg["backbone"])
    cases = load_cases(args.data_dir, [d.strip() for d in args.domains.split(",")],
                       args.max_cases, args.seed, per_domain=args.per_domain)
    print(f"{len(cases)} training cases", flush=True)
    if not cases:
        raise SystemExit("[distill] 0 training cases - check --data-dir/--domains; "
                         "each domain needs {data_dir}/{domain}/train.jsonl "
                         "(build with: python -m oev.convert && python -m oev.convert_banking77 "
                         "&& python -m oev.convert_typed)")

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    for epoch in range(args.epochs):
        tl = run_epoch(student, teachers, cases, packer, device, opt, scaler, args, epoch, log)
        msg = f"epoch {epoch} done - train loss {tl:.4f}"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
        torch.save({"config": student.cfg, "state": student.state_dict()},
                   os.path.join(args.out, "student.pt"))
        print(f"saved -> {args.out}/student.pt", flush=True)


if __name__ == "__main__":
    main()
