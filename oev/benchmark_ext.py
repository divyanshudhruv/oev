"""Extended benchmark metrics for typed-decisions: soft accuracy, Brier score,
score MAE, and per-workflow / per-primitive breakdowns.

Mirrors the metric set published in Laya's typed-decisions table so results
are directly comparable column-for-column.

Usage:
    python -m oev.benchmark_ext --checkpoint CKPT --data-dir data/typed
Ensembles:
    python -m oev.benchmark_ext --ckpts a.pt,b.pt,c.pt --data-dir data/typed
"""

import argparse
import json

import torch
import torch.nn.functional as F

from oev.evaluate import load_model
from oev.tokenizer_hf import HFTokenPacker


def _predict_probs(model, packer, state, question, device, gamma=1.0):
    pq = {
        "name": question["name"],
        "type": question["type"],
        "instructions": question.get("instructions", question["type"]),
        "options": question["options"],
        "answer": question["answer"],
    }
    ids, anchors, label = packer.pack(state, pq, model.cfg["max_len"])
    tids = torch.tensor([ids], device=device)
    pmask = torch.zeros(1, len(ids), dtype=torch.bool, device=device)
    apos = torch.tensor([anchors], device=device)
    with torch.no_grad():
        logits = model(tids, pmask, apos)
    probs = torch.softmax(logits[0].float(), dim=-1)
    if gamma != 1.0:
        probs = probs.clamp_min(1e-9) ** gamma
        probs = probs / probs.sum()
    return probs, label, question["type"], question.get("target")


def ece_metric(confs, corrs, n_bins=10):
    """Expected Calibration Error: weighted |confidence - accuracy| over bins."""
    if not confs:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for c, o in zip(confs, corrs):
        b = min(int(c * n_bins), n_bins - 1)
        bins[b].append((c, o))
    e = 0.0
    for b in bins:
        if not b:
            continue
        acc = sum(o for _, o in b) / len(b)
        conf = sum(c for c, _ in b) / len(b)
        e += (len(b) / len(confs)) * abs(conf - acc)
    return e


def confident_error_rate(confs, corrs, threshold=0.9):
    """Fraction of all answers that are wrong despite p >= threshold."""
    errs = [(c, o) for c, o in zip(confs, corrs) if c >= threshold]
    if not errs:
        return 0.0, 0
    return sum(1 - o for _, o in errs) / len(confs), len(errs)


def coverage_at_error_budget(confs, corrs, budget=0.05):
    """Largest fraction of questions automatable (answer taken when p >= t)
    while keeping the error rate on automated answers <= budget."""
    best = 0.0
    for t in sorted(set(confs), reverse=True):
        kept = [(c, o) for c, o in zip(confs, corrs) if c >= t]
        if not kept:
            continue
        err = sum(1 - o for _, o in kept) / len(kept)
        cov = len(kept) / len(confs)
        if err <= budget:
            best = max(best, cov)
    return best


def aurc(confs, corrs):
    """Area under the risk-coverage curve: sort by confidence desc,
    risk at coverage c is the error rate of the top-c fraction. Lower is better."""
    pairs = sorted(zip(confs, corrs), key=lambda x: -x[0])
    n = len(pairs)
    cum_err = 0.0
    total = 0.0
    for i, (_, o) in enumerate(pairs):
        cum_err += 1 - o
        total += cum_err / (i + 1)
    return total / n


def evaluate_metrics(checkpoints, data_dir, device="cuda", gamma=1.0, latency=False, permute=0):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    models = [load_model(c, device) for c in checkpoints]
    packers = [HFTokenPacker(m.cfg["backbone"]) for m in models]

    rows = [json.loads(l) for l in open(f"{data_dir}/test.jsonl", encoding="utf-8")]

    n = correct = 0
    soft_acc_sum = 0.0
    brier_sum = 0.0
    score_mae_sum = 0.0
    score_n = 0
    confs = []
    corrs = []
    by_type = {}
    by_domain = {}

    with torch.no_grad():
        for r in rows:
            for q in r["questions"]:
                probs_sum = None
                label = None
                target = None
                qtype = None
                for m, p in zip(models, packers):
                    probs, label, qtype, target = _predict_probs(m, p, r["state"], q, device, gamma=gamma)
                    probs_sum = probs if probs_sum is None else probs_sum + probs
                probs = probs_sum / len(models)

                n += 1
                hit = int(probs.argmax().item() == label)
                correct += hit
                confs.append(probs.max().item())
                corrs.append(hit)

                # soft accuracy: the probability mass the model put on the gold answer
                soft_acc_sum += probs[label].item()

                # Brier score against the gold one-hot (lower is better)
                oh = torch.zeros_like(probs)
                oh[label] = 1.0
                brier_sum += ((probs - oh) ** 2).sum().item()

                # score MAE: |expected level - gold level| for score questions
                if qtype == "score":
                    levels = torch.arange(probs.numel(), dtype=torch.float32, device=probs.device)
                    ev = (probs * levels).sum().item()
                    score_mae_sum += abs(ev - label)
                    score_n += 1

                agg = by_type.setdefault(qtype, [0, 0, 0.0])
                agg[0] += hit
                agg[1] += 1
                agg[2] += probs[label].item()

                agg = by_domain.setdefault(r.get("domain", "typed"), [0, 0, 0.0])
                agg[0] += hit
                agg[1] += 1
                agg[2] += probs[label].item()

    conf_err, conf_err_n = confident_error_rate(confs, corrs)
    conf_err_wrong = sum(1 - o for c, o in zip(confs, corrs) if c >= 0.9)
    result = {
        "accuracy": correct / n,
        "soft_acc": soft_acc_sum / n,
        "brier": brier_sum / n,
        "score_mae": (score_mae_sum / score_n) if score_n else None,
        "ece": ece_metric(confs, corrs),
        "confident_errors": conf_err,
        "coverage_at_5pct": coverage_at_error_budget(confs, corrs, 0.05),
        "aurc": aurc(confs, corrs),
        "n": n,
    }
    print(f"accuracy : {result['accuracy']:.4f}")
    print(f"soft acc : {result['soft_acc']:.4f}")
    print(f"brier    : {result['brier']:.4f}")
    if result["score_mae"] is not None:
        print(f"score MAE: {result['score_mae']:.4f}")
    print(f"ece      : {result['ece']:.4f}")
    print(f"conf err : {result['confident_errors']:.4f}  ({conf_err_wrong} of {conf_err_n} answers at p>=0.9 are wrong)")
    print(f"coverage : {result['coverage_at_5pct']:.4f}  (automatable at <=5% error)")
    print(f"aurc     : {result['aurc']:.4f}")
    print(f"n        : {result['n']}")

    print("\nper primitive:")
    for t, (c, tot, s) in sorted(by_type.items()):
        print(f"  {t:<7} acc {c / tot:.4f}  soft acc {s / tot:.4f}  (n={tot})")
    print("per workflow:")
    for d, (c, tot, s) in sorted(by_domain.items()):
        print(f"  {d:<28} acc {c / tot:.4f}  soft acc {s / tot:.4f}  (n={tot})")

    if permute > 1:
        permute_flip_rate(models[0], packers[0], rows, device, permute)
    if latency:
        measure_latency(models[0], packers[0], rows, device)
    return result


def permute_flip_rate(model, packer, rows, device, n_perm=6, max_cases=300):
    """How much does answer identity depend on option order? For choice
    questions, rotate the options n_perm times and count how often the
    argmax changes relative to the identity order. Lower is better."""
    import itertools

    cases = []
    for r in rows:
        for q in r["questions"]:
            if q["type"] == "choice" and len(q["options"]) <= 24:
                cases.append((r["state"], q))
            if len(cases) >= max_cases:
                break
        if len(cases) >= max_cases:
            break
    if not cases:
        print("permute: no choice questions found")
        return

    def predict(s, q):
        ids, an, _ = packer.pack(s, q, model.cfg["max_len"])
        with torch.no_grad():
            logits = model(torch.tensor([ids], device=device),
                           torch.zeros(1, len(ids), dtype=torch.bool, device=device),
                           torch.tensor([an], device=device))
        return logits[0].argmax().item()

    flips = 0
    total = 0
    for s, q in cases:
        base_pick = q["options"][predict(s, q)]
        for k in range(1, n_perm):
            rot = q["options"][k:] + q["options"][:k]
            rq = dict(q, options=rot)
            pick = rot[predict(s, rq)]
            total += 1
            if pick != base_pick:
                flips += 1
    rate = flips / total if total else 0.0
    print(f"\npermute: {flips}/{total} answer changes under {n_perm} option rotations (flip rate {rate:.4f})")


def measure_latency(model, packer, rows, device, n_single=50, n_batch=200, batch_size=32):
    """Latency protocol: p50 single-question and batched per-question throughput.
    Model must already be on `device` and warmed up by at least a few calls."""
    import time

    qs = [(r["state"], q) for r in rows for q in r["questions"]]

    def run_one(s, q):
        pq = {"name": q["name"], "type": q["type"], "instructions": q.get("instructions", q["type"]), "options": q["options"], "answer": q["answer"]}
        ids, an, _ = packer.pack(s, pq, model.cfg["max_len"])
        t0 = time.perf_counter()
        with torch.no_grad():
            model(torch.tensor([ids], device=device), torch.zeros(1, len(ids), dtype=torch.bool, device=device), torch.tensor([an], device=device))
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000

    for s, q in qs[:3]:  # warmup
        run_one(s, q)
    times = sorted(run_one(s, q) for s, q in qs[:n_single])
    print(f"\nlatency single-question p50: {times[len(times) // 2]:.1f} ms  (n={n_single}, warmup 3)")

    # batched throughput: pack n_batch questions, pad to common length, one forward per batch
    items = qs[:n_batch]
    t0 = time.perf_counter()
    done = 0
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        packed = []
        for s, q in chunk:
            pq = {"name": q["name"], "type": q["type"], "instructions": q.get("instructions", q["type"]), "options": q["options"], "answer": q["answer"]}
            packed.append(packer.pack(s, pq, model.cfg["max_len"]))
        L = max(len(p[0]) for p in packed)
        A = max(len(p[1]) for p in packed)
        B = len(packed)
        ids = torch.zeros(B, L, dtype=torch.long, device=device)
        pmask = torch.ones(B, L, dtype=torch.bool, device=device)
        apos = torch.zeros(B, A, dtype=torch.long, device=device)
        for j, (pid, pan, _) in enumerate(packed):
            ids[j, : len(pid)] = torch.tensor(pid, device=device)
            pmask[j, : len(pid)] = False
            apos[j, : len(pan)] = torch.tensor(pan, device=device)
        with torch.no_grad():
            model(ids, pmask, apos)
        done += B
    if device == "cuda":
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000
    print(f"latency batched: {total_ms / done:.1f} ms/question  (n={done}, batch={batch_size})")
    print(f"throughput: {done / (total_ms / 1000):.0f} questions/sec")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--ckpts", default=None, help="comma-separated checkpoints (ensemble)")
    p.add_argument("--data-dir", default="data/typed")
    p.add_argument("--sharpen", type=float, default=1.0, help="confidence exponent gamma; >1 sharpens distributions")
    p.add_argument("--latency", action="store_true", help="also measure single p50 and batched throughput")
    p.add_argument("--permute", type=int, default=0, help="also run the option-order sensitivity check with N rotations (e.g. 6)")
    args = p.parse_args()
    ckpts = ([c.strip() for c in args.ckpts.split(",") if c.strip()]
             if args.ckpts else [args.checkpoint])
    evaluate_metrics(ckpts, args.data_dir, gamma=args.sharpen, latency=args.latency, permute=args.permute)
