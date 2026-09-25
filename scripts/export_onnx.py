import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

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
        rows = [  # synthetic fallback so the pipeline can be proven on any machine
            {"id": f"syn-{i}", "state": "A short synthetic state string for packing.",
             "questions": [{"name": "topic", "type": "choice", "instructions": "Pick a topic.",
                            "options": ["world", "sports", "business", "sci/tech"],
                            "answer": "world"}]}
            for i in range(5)
        ]
    return rows


def question_for(q):
    return {"name": q["name"], "type": q["type"],
            "instructions": q.get("instructions", q["type"]),
            "options": q["options"], "answer": q["answer"]}


def pack_case(packer, state, q, max_len):
    ids, anchors, _ = packer.pack(state, question_for(q), max_len)
    return (torch.tensor([ids]).long(),
            torch.zeros(1, len(ids), dtype=torch.bool),
            torch.tensor([anchors]).long())


def ort_output(sess, ids, pmask, apos):
    return sess.run(None, {"ids": ids.numpy(), "pad_mask": pmask.numpy(), "anchor_pos": apos.numpy()})[0]


def maxdiff(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max())


def onnx_size(path):
    total = os.path.getsize(path)
    data = path + ".data"
    if os.path.exists(data):
        total += os.path.getsize(data)
    return total / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="data/typed")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-cases", type=int, default=20)
    args = ap.parse_args()

    ck = os.path.basename(args.checkpoint).replace(".pt", "")
    out_dir = args.out_dir or os.path.join("checkpoints_export")
    os.makedirs(out_dir, exist_ok=True)
    fp32_path = os.path.join(out_dir, f"{ck}.onnx")
    int8_path = os.path.join(out_dir, f"{ck}.int8.onnx")

    model = load_model(args.checkpoint, "cpu")
    max_len = model.cfg["max_len"]
    packer = HFTokenPacker(model.cfg["backbone"])
    rows = load_rows(args.data_dir, args.max_cases)
    q0 = rows[0]["questions"][0]
    ids, pmask, apos = pack_case(packer, rows[0]["state"], q0, max_len)
    print(f"example packed: seq_len={ids.shape[1]} anchors={apos.shape[1]} max_len={max_len}")

    with torch.no_grad():
        torch_logits = model(ids, pmask, apos).numpy()

    t0 = time.time()
    torch.onnx.export(
        model, (ids, pmask, apos), fp32_path,
        input_names=["ids", "pad_mask", "anchor_pos"],
        output_names=["logits"],
        dynamic_axes={"ids": {0: "batch", 1: "seq"}, "pad_mask": {0: "batch", 1: "seq"},
                      "anchor_pos": {0: "batch", 1: "anchors"}, "logits": {0: "batch", 1: "anchors"}},
        opset_version=17,
    )
    print(f"fp32 export: {time.time()-t0:.1f}s -> {fp32_path}")

    sess32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    far = maxdiff(torch_logits, ort_output(sess32, ids, pmask, apos))
    print(f"fp32 onnx vs torch: maxdiff {far:.6f}  ({'PASS' if far < 1e-3 else 'FAIL'})")

    verified = []
    for r in rows[:10]:
        for q in r["questions"]:
            i2, m2, a2 = pack_case(packer, r["state"], q, max_len)
            with torch.no_grad():
                t = model(i2, m2, a2).numpy()
            o32 = ort_output(sess32, i2, m2, a2)
            d = maxdiff(t, o32)
            verified.append(d)
            if d >= 1e-3:
                break
    print(f"fp32 onnx vs torch over {len(verified)} cases: maxdiff {max(verified):.6f}")

    t0 = time.time()
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
    print(f"int8 quantize: {time.time()-t0:.1f}s -> {int8_path}")

    sess8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    i8far = maxdiff(torch_logits, ort_output(sess8, ids, pmask, apos))
    print(f"int8 vs fp32 example: maxdiff {i8far:.6f}  ({'PASS' if i8far < 0.05 else 'CHECK'})")

    print(f"\nsizes: fp32 {onnx_size(fp32_path):.1f} MB | "
          f"int8 {onnx_size(int8_path):.1f} MB")


if __name__ == "__main__":
    main()