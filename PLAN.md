# OEV research plan

Working plan for the next experiments: the criteria each run must meet, and
what past runs produced. Results land in [BENCHMARKS.md](BENCHMARKS.md) and
forward work lives in [ROADMAP.md](ROADMAP.md).

## Where we stand

Shipped: the distilled generalist `student-r2b-oev-tiny.pt` (typed `0.6480`,
Banking77 `0.7964`, emotion zero-shot `0.6505`), typed specialist `0.7705`
single / `0.7760` ensemble, Banking77 soup `0.8584`, AG News `0.9489`,
DAIR Emotion `0.9300`, `22.2 ms` p50 on T4. Weights on Hugging Face, package
on PyPI, demo Space live.

> [!WARNING]
> Gamma `2.5` was selected from an evaluation sweep, and the validation-only
> selection log is not tracked here. Treat the sharpened rows as exploratory
> until the selection log and raw measurements are archived.

## Recorded failures (do not repeat)

- **gold-CE + teacher-KL distillation collapses to uniform**: round 2 added
  the gold answer cross-entropy to the KL loss and the student flatlined at
  b77 `0.0104`, below random. Round 2b (pure KL, per-domain balanced data)
  rescued it. Keep distillation pure-KL unless a new loss is gated on a
  small validation run first.
- **from-scratch backbone runs can stall at the uniform plateau**: two
  Banking77 runs pinned at ln(77) for 1500+ steps with the standard
  2e-5/1e-3 param groups. Warm-starting from a released checkpoint avoids
  it entirely. Open question: is the head LR or schedule at fault?
- **validation OOMs at end of epoch 0** when the valid pass runs in fp32
  with autograd graphs (fixed in run_epoch: no_grad + autocast always).
- **batch 16 OOMs the T4** at ~570 packed tokens with 77 anchors. Batch 8
  is the ceiling for high-cardinality work on 16 GB.
- **anchor positions must be clamped after sequence clipping**: `pack()`
  records anchors before the `max_len` clip; when a question's fixed
  overhead exceeds the budget, stale anchors point past the sequence and
  the encoder's gather raises a device-side assert. Fixed in
  `tokenizer_hf.py`; any future packer change must clamp post-clip.
- **gamma selected by inspecting evaluation sweeps** produced the
  sharpened rows; they are labeled exploratory in BENCHMARKS.md. Future
  sharpening is selected on the valid split only.

## Adoption criteria for the next runs

- **round 3 (6-teacher distillation with NLI)**: adopt the new student as
  the generalist only if it keeps typed >= `0.60`, b77 >= `0.77`, emotion
  >= `0.63` and lifts WANLI >= `0.50`. Otherwise r2b stays shipped and the
  MNLI specialist ships as a separate artifact.
- **Banking77 ensemble of four (soup + re-tune + a + b)**: publish if it
  clears the 3-member `0.8529`; the Jev `0.870` gap closes only above that.
- **round 4 (per-domain specialist teachers)**: target is one file near
  specialist numbers everywhere: typed ~`0.70+`, b77 ~`0.83`, NLI `0.55+`.

## Backlog (unordered, uncommitted)

- permute + coverage numbers for the published rows (code exists; GPU pass)
- additional domain specialists (guardrails, moderation, RAG filtering)
