"""Extended benchmark metrics for typed-decisions, mirroring Laya's published
table column-for-column (soft accuracy, Brier, ECE, coverage, AURC)."""

import argparse
import json
import os
from datetime import datetime, timezone

import torch

from oev.evaluate import ece, load_model, pack_question
from oev.tokenizer_hf import HFTokenPacker


def _predict_probs(model, packer, state, question, device, gamma=1.0):
    pq = pack_question(question)
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


# the published metric name keeps its old import surface and its 10-bin
# default; the implementation lives in oev.evaluate.ece (the canonical copy,
# whose own default of 15 bins serves oev.benchmark's published rows)


def ece_metric(confs, corrs, n_bins=10):
    return ece(confs, corrs, bins=n_bins)


def confident_error_rate(confs, corrs, threshold=0.9):
    # wrong answers with p >= threshold, as a fraction of ALL answers
    errs = [(c, o) for c, o in zip(confs, corrs) if c >= threshold]
    if not errs:
        return 0.0, 0
    return sum(1 - o for _, o in errs) / len(confs), len(errs)


def coverage_at_error_budget(confs, corrs, budget=0.05):
    # largest fraction automatable (p >= t) while automated error stays <= budget
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
    # area under the risk-coverage curve, lower is better
    pairs = sorted(zip(confs, corrs), key=lambda x: -x[0])
    n = len(pairs)
    cum_err = 0.0
    total = 0.0
    for i, (_, o) in enumerate(pairs):
        cum_err += 1 - o
        total += cum_err / (i + 1)
    return total / n


def resolve_device(device="cuda", allow_cpu=False):
    # fail loudly on a missing GPU unless allow_cpu: CPU numbers must never
    # pass for GPU numbers in a published report
    if device.startswith("cuda") and not torch.cuda.is_available():
        if allow_cpu:
            print("WARNING: CUDA unavailable; running on CPU (--allow-cpu). "
                  "Latency numbers are not comparable to GPU runs.")
            return "cpu"
        raise SystemExit(
            "CUDA requested but unavailable. Re-run with --allow-cpu "
            "if CPU numbers are what you want.")
    return device


def write_manifest(out_dir, checkpoint, data_dir, device, result):
    # per-run receipt: exact inputs, resolved device, metrics. published
    # numbers should be reproducible from these
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = os.path.splitext(os.path.basename(str(checkpoint)))[0] or "ensemble"
    path = os.path.join(out_dir, f"{stamp}-{name}.json")
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": checkpoint,
        "data_dir": data_dir,
        "device": device,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "metrics": result,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"manifest: {path}")
    return path


def evaluate_metrics(checkpoints, data_dir, device="cuda", gamma=1.0, latency=False,
                     permute=0, allow_cpu=False, manifest_dir=None):
    device = resolve_device(device, allow_cpu)
    models = []
    for i, ckpt in enumerate(checkpoints, 1):
        # every load is a full backbone init: announce it so the notebook log
        # never shows an unexplained silent stretch
        print(f"loading model {i}/{len(checkpoints)}: {ckpt} (model load, 1-2 min)", flush=True)
        models.append(load_model(ckpt, device))
    packers = [HFTokenPacker(m.cfg["backbone"]) for m in models]

    with open(f"{data_dir}/test.jsonl", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]

    n = correct = 0
    soft_acc_sum = 0.0
    brier_sum = 0.0
    score_mae_sum = 0.0
    score_n = 0
    confs = []
    corrs = []
    by_type = {}
    by_domain = {}

    import time
    t_eval = time.time()
    with torch.no_grad():
        for r in rows:
            for q in r["questions"]:
                probs_sum = None
                label = None
                qtype = None
                for m, p in zip(models, packers):
                    probs, label, qtype, _target = _predict_probs(m, p, r["state"], q, device, gamma=gamma)
                    probs_sum = probs if probs_sum is None else probs_sum + probs
                probs = probs_sum / len(models)

                n += 1
                hit = int(probs.argmax().item() == label)
                correct += hit
                confs.append(probs.max().item())
                corrs.append(hit)
                if n % 500 == 0:
                    print(f"eval progress: {n} questions ({n / (time.time() - t_eval):.0f} q/s)", flush=True)

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
        "device": device,
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
    print(f"device   : {device}")
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
        result["latency"] = measure_latency(models[0], packers[0], rows, device)
    if manifest_dir:
        ckpt_label = checkpoints[0] if len(checkpoints) == 1 else ",".join(checkpoints)
        write_manifest(manifest_dir, ckpt_label, data_dir, device, result)
    return result


def permute_flip_rate(model, packer, rows, device, n_perm=6, max_cases=300):
    # rotate choice options n_perm times, count argmax changes vs the
    # identity order. lower is better

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
    # p50 single-question and batched per-question ms; model must already be
    # on the device and warmed up
    import time

    qs = [(r["state"], q) for r in rows for q in r["questions"]]

    def run_one(s, q):
        pq = pack_question(q)
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
            packed.append(packer.pack(s, pack_question(q), model.cfg["max_len"]))
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
    return {
        "single_p50_ms": round(times[len(times) // 2], 2),
        "single_ms": [round(t, 2) for t in times],
        "batched_ms_per_q": round(total_ms / done, 2),
        "throughput_qps": round(done / (total_ms / 1000), 1),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--ckpts", default=None, help="comma-separated checkpoints (ensemble)")
    p.add_argument("--data-dir", default="data/typed")
    p.add_argument("--sharpen", type=float, default=1.0, help="confidence exponent gamma; >1 sharpens distributions")
    p.add_argument("--latency", action="store_true", help="also measure single p50 and batched throughput")
    p.add_argument("--permute", type=int, default=0, help="also run the option-order sensitivity check with N rotations (e.g. 6)")
    p.add_argument("--device", default="cuda", help="cuda or cpu; a missing GPU fails unless --allow-cpu is set")
    p.add_argument("--allow-cpu", action="store_true", help="fall back to CPU when CUDA is requested but unavailable")
    p.add_argument("--manifest-dir", default="runs", help="write a per-run receipt JSON here (empty string disables)")
    args = p.parse_args()
    ckpts = ([c.strip() for c in args.ckpts.split(",") if c.strip()]
             if args.ckpts else [args.checkpoint])
    evaluate_metrics(ckpts, args.data_dir, device=args.device, gamma=args.sharpen,
                     latency=args.latency, permute=args.permute, allow_cpu=args.allow_cpu,
                     manifest_dir=args.manifest_dir or None)
