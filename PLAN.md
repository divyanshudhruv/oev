# OEV research plan

Living plan: what runs next, the criteria decided before each run, and what
came of past runs. Every result, met or not, is recorded in this file.

## Where we stand

Published: typed-decisions 0.7705 single / 0.7760 ensemble (ECE 0.0298
sharpened), Banking77 0.8303 (ECE 0.1129 at gamma 1.5), AG News 0.9489,
DAIR Emotion 0.9300, 22.2 ms p50 on T4. Weights on Hugging Face, package
on PyPI, demo Space live.

## Recorded failures (do not repeat)

- **from-scratch backbone runs can stall at the uniform plateau**: two
  Banking77 runs pinned at ln(77) for 1500+ steps with the standard
  2e-5/1e-3 param groups. Warm-starting from a released checkpoint avoids
  it entirely. Open question: is the head LR or schedule at fault?
- **validation OOMs at end of epoch 0** when the valid pass runs in fp32
  with autograd graphs (fixed in run_epoch: no_grad + autocast always).
- **batch 16 OOMs the T4** at ~570 packed tokens with 77 anchors. Batch 8
  is the ceiling for high-cardinality work on 16 GB.
- **gamma selected by inspecting evaluation sweeps** produced the
  sharpened rows; they are labeled exploratory in BENCHMARKS.md. Future
  sharpening is selected on the valid split only.

## Round 1 - distillation (next GPU session, ~5h)

One student model that keeps the ensemble's typed accuracy and the
multi-task checkpoint's general skills.

- teachers: typed ensemble (td5 + rlcd + rlcd-soup + rlcd-s1) on typed
  data; mt checkpoint on ag_news + emotion data
- student: fresh 184M backbone, soft targets throughout
- **adopt if**: student >= 0.765 on typed-decisions test (within 1.1 pp of
  the ensemble) AND beats td5 zero-shot on the round-2 OOD suite AND
  ECE <= 0.10. Otherwise record the deltas and keep the ensemble shipped.

## Round 2 - out-of-domain evaluation (can precede round 1)

First published OOD row. Suites: WANLI (public, HF) converted to choice
questions; 1,000-case locked read.

- models evaluated: td5, mt, and the round-1 student if it exists
- **adopt if**: informational only - every number gets published,
  including bad ones. A number for the empty OOD cell is the deliverable.

## Round 3 - shipped-calibrated checkpoints

Fold temperature fitting into train.py so every new checkpoint stores a
valid-split-fitted temperature applied at load.

- **adopt if**: ECE on typed-decisions test improves or holds vs the
  current as-published numbers, with accuracy unchanged. Retrofitting the
  six existing checkpoints is explicitly rejected: re-calibrating and
  re-uploading 4.4 GB to change metadata is not worth it; the next
  generation ships calibrated instead.

## Backlog (unordered, uncommitted)

- permute + coverage numbers for the published rows (code exists; GPU pass)
- CPU latency characterization
- option-order and question-isolation parity documented in BENCHMARKS.md
- ONNX / INT8 export for CPU deployment
- additional domain specialists (guardrails, moderation, RAG filtering)
