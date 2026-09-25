import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import onnxruntime as ort
import torch
from onnxruntime.capi.onnxruntime_pybind11_state import Fail

from oev.evaluate import load_model
from oev.tokenizer_hf import HFTokenPacker


def load_rows(data_dir, max_cases):
    rows = []
    p = os.path.join(data_dir, "test.jsonl") if data_dir else ""
    if p and os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
                if max_cases and len(rows) >= max_cases:
                    break
    if not rows:
        rows = [{"id": f"syn-{i}", "state": "Synthetic state text for a latency bench.",
                 "questions": [{"name": "topic", "type": "choice", "instructions": "Pick.",
                                "options": ["a", "b", "c", "d"], "answer": "a"}]}
                for i in range(100)]
    return rows


def question_for(q):
    return {"name": q["name"], "type": q["type"],
            "instructions": q.get("instructions", q["type"]),
            "options": q["options"], "answer": q["answer"]}


def bench(forward, packer, model_cfg, rows, n_single, n_batch, batch_size):
    qs = [(r["state"], q) for r in rows for q in r["questions"]]
    if not qs:
        raise SystemExit("no questions found to benchmark")
    max_len = model_cfg["max_len"]

    def run_one(s, q):
        ids, anchors, _ = packer.pack(s, question_for(q), max_len)
        ids = torch.tensor([ids]).long()
        pmask = torch.zeros(1, ids.shape[1], dtype=torch.bool)
        apos = torch.tensor([anchors]).long()
        t0 = time.perf_counter()
        forward(ids, pmask, apos)
        return (time.perf_counter() - t0) * 1000

    for s, q in qs[:3]:
        run_one(s, q)  # warmup
    times = sorted(run_one(s, q) for s, q in qs[:n_single])
    p50 = times[len(times) // 2]

    items = qs[:n_batch]
    packed = []
    for s, q in items:
        ids, anchors, _ = packer.pack(s, question_for(q), max_len)
        packed.append((ids, anchors))
    t0 = time.perf_counter()
    done = 0
    for i in range(0, len(packed), batch_size):
        chunk = packed[i:i + batch_size]
        L = max(len(p[0]) for p in chunk)
        A = max(len(p[1]) for p in chunk)
        B = len(chunk)
        ids = torch.zeros(B, L, dtype=torch.long)
        pmask = torch.ones(B, L, dtype=torch.bool)
        apos = torch.zeros(B, A, dtype=torch.long)
        for j, (pid, pan) in enumerate(chunk):
            ids[j, : len(pid)] = torch.tensor(pid)
            pmask[j, : len(pid)] = False
            apos[j, : len(pan)] = torch.tensor(pan)
        forward(ids, pmask, apos)
        done += B
    total_ms = (time.perf_counter() - t0) * 1000
    return p50, total_ms / done, done / max(total_ms / 1000, 1e-9), times


def mb(path):
    total = os.path.getsize(path)
    data = path + ".data"
    if os.path.exists(data):
        total += os.path.getsize(data)
    return total / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="data/banking77")
    ap.add_argument("--out-dir", default="checkpoints_export")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--compile", action="store_true",
                    help="opt-in: torch.compile backend (slow first pass on CPU)")
    ap.add_argument("--n-single", type=int, default=20)
    ap.add_argument("--n-batch", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    model = load_model(args.checkpoint, "cpu")
    packer = HFTokenPacker(model.cfg["backbone"])
    rows = load_rows(args.data_dir, max(args.n_single, args.n_batch) + 10)

    ck = os.path.basename(args.checkpoint).replace(".pt", "")
    fp32_path = os.path.join(args.out_dir, f"{ck}.onnx")
    int8_path = os.path.join(args.out_dir, f"{ck}.int8.onnx")

    # auto-export when the ONNX artifacts are missing so one command does it all
    if not os.path.exists(fp32_path) or not os.path.exists(int8_path):
        print("onnx artifacts missing -> running scripts/export_onnx.py first")
        r = subprocess.run(
            [sys.executable, "scripts/export_onnx.py",
             "--checkpoint", args.checkpoint, "--data-dir", args.data_dir,
             "--out-dir", args.out_dir],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))) or None,
            check=True,
        )
        if r.returncode != 0:
            print("export failed; benchmarking torch only")

    results = []  # (name, p50_ms, batched_ms_per_q, qps, size_mb, single_samples_ms)
    with torch.no_grad():
        p50, bq, qps, samples = bench(model.forward, packer, model.cfg, rows,
                                      args.n_single, args.n_batch, args.batch_size)
        results.append(("fp32 torch", p50, bq, qps,
                        os.path.getsize(args.checkpoint) / 1e6, samples))
        print(f"fp32 torch   : p50 {p50:6.1f} ms | batched {bq:6.1f} ms/q | {qps:5.0f} q/s")

        if args.compile:
            try:
                compiled = torch.compile(model)
                p50, bq, qps, samples = bench(compiled.forward, packer, model.cfg, rows,
                                              args.n_single, args.n_batch, args.batch_size)
                results.append(("fp32 torch.compile", p50, bq, qps,
                                os.path.getsize(args.checkpoint) / 1e6, samples))
                print(f"fp32 compile : p50 {p50:6.1f} ms | batched {bq:6.1f} ms/q | {qps:5.0f} q/s")
            except (RuntimeError, IndentationError, ValueError) as e:
                print(f"torch.compile skipped: {e}")

    sess32 = sess8 = None
    if os.path.exists(fp32_path):
        try:
            sess32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
        except (RuntimeError, Fail) as e:
            print(f"onnx fp32 skipped (session load failed): {e}")
        if sess32 is not None:
            p50, bq, qps, samples = bench(
                lambda i, m, a: sess32.run(None, {"ids": i.numpy(), "pad_mask": m.numpy(),
                                                  "anchor_pos": a.numpy()}),
                packer, model.cfg, rows, args.n_single, args.n_batch, args.batch_size)
            results.append(("onnx fp32", p50, bq, qps, mb(fp32_path), samples))
            print(f"onnx fp32    : p50 {p50:6.1f} ms | batched {bq:6.1f} ms/q | {qps:5.0f} q/s")
    if os.path.exists(int8_path):
        try:
            sess8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
        except (RuntimeError, Fail) as e:
            print(f"onnx int8 skipped (session load failed): {e}")
        if sess8 is not None:
            p50, bq, qps, samples = bench(
                lambda i, m, a: sess8.run(None, {"ids": i.numpy(), "pad_mask": m.numpy(),
                                                 "anchor_pos": a.numpy()}),
                packer, model.cfg, rows, args.n_single, args.n_batch, args.batch_size)
            results.append(("onnx int8", p50, bq, qps, mb(int8_path), samples))
            print(f"onnx int8    : p50 {p50:6.1f} ms | batched {bq:6.1f} ms/q | {qps:5.0f} q/s")

    # archive a receipt so published numbers carry their raw samples
    ts = time.strftime("%Y%m%d-%H%M%S")
    runs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    ck_hash = hashlib.sha256()
    with open(args.checkpoint, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            ck_hash.update(chunk)
    manifest = {
        "kind": "latency-bench",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": ck_hash.hexdigest(),
        "device": "cpu",
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "args": {"data_dir": args.data_dir, "n_single": args.n_single,
                 "n_batch": args.n_batch, "batch_size": args.batch_size},
        "backends": {
            name: {"p50_ms": round(p50, 2), "batched_ms_per_q": round(bq, 2),
                   "qps": round(qps, 1), "size_mb": round(size, 1),
                   "samples_ms": [round(s, 2) for s in samples]}
            for name, p50, bq, qps, size, samples in results
        },
    }
    manifest_path = os.path.join(runs_dir, f"{ts}-latency-{ck}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest -> {manifest_path}")

    print(f"\n{'backend':<14}{'p50(ms)':>10}{'batch ms/q':>12}{'q/s':>8}{'MB':>9}")
    for name, p50, bq, qps, size, samples in results:
        print(f"{name:<14}{p50:>10.1f}{bq:>12.1f}{qps:>8.0f}{size:>9.1f}")


if __name__ == "__main__":
    main()
