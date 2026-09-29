# OEV roadmap

Measured results go to [BENCHMARKS.md](BENCHMARKS.md). Done items are checked with the number they landed.

## Near term

- [x] Round 3 distillation with 3 teachers and 5 domains, MNLI included. Pure-KL loss only, because round 2 already showed what gold-CE does to the student. Landed typed `0.6895` (r2b `0.6480`), Banking77 `0.8370` (r2b `0.7964`), emotion zero-shot `0.8650` (r2b `0.6505`); checkpoint published as `student-r3-oev-tiny.pt`
- [x] Try the 4-member Banking77 ensemble. Landed `0.8594` (soup4, +0.10 over the 3-member soup); the 5-member attempt showed ensemble saturation, so past Jev's `0.870` needs a different lever (lineage diversity, data augmentation or distillation), not more warm-starts
- [x] Rebuild the MNLI specialist. The rebuild landed WANLI `0.5265` (same recipe as the lost `0.5645` original; receipt in `runs/`), and the Hub file `mnli-oev-tiny.pt` now holds the real specialist instead of multi-task weights

## Model quality

- [x] Round 4 distillation: 10 domains, the live head-to-head fix sets mixed in. Landed the best generalist typed `0.6985` and emotion zero-shot `0.8700`, trading Banking77 to `0.8188` (the fix domains are not b77 classes). Published as `student-r4-oev-tiny.pt`. The fix-domain gains await a Kev head-to-head validation. Fix datasets that fed it:
  - [ ] invoices and billing words appearing in technical contexts (the model hears "invoice" and ignores the 500 error)
  - [ ] multi-issue tickets with a "both" option, so two problems in one message stop collapsing to the first noun
  - [ ] benign security logins next to real incidents, to kill the alert-everything lean
  - [ ] sarcastic praise labeled very negative
  - [ ] trace pairs where the only difference is runtime, one alerting and one silent, so latency stops being invisible
  - [ ] a head-consistency term so "approve" never pairs with "clear mismatch"
- [ ] Bring coherence back up. Distillation traded decision conservatism for breadth, so the student scores `3/5` on the five decision-coherence scenarios where the teacher scores `4/5`. A conservative-decision term in the loss should close that
- [x] Fine-tune on ANLI R1's train split. Landed `0.5750` in-domain (chance `0.333`) warm-started from the rebuilt specialist, and the adversarial tune lifted WANLI transfer to `0.5690` (from `0.5265`). OOD overconfidence (conf-err `0.1290`) is the open cost
- [ ] Train more specialists for guardrails, moderation and RAG filtering

## Inference

- [x] Run the INT8 ONNX bench. It landed at `54.2 ms` p50 on 8 threads, about 4.7x under the torch fp32 figure on the same machine, and the artifact is `228 MB` instead of `735 MB`
- [x] Try fp16 autocast on the eval path. Accuracy holds exactly (0.7705 both dtypes on the full test set); single-question p50 landed at 28.89 ms vs 32.61 ms fp32 on the same T4 instance (11 percent at batch 1, latency-bound). Receipt in `runs/`
- [x] Measure the batched-latency bound for shared-state encoding: 4 questions in one padded batch run 72.17 ms vs 98.54 ms per-question, a 27 percent cut any true one-pass encoder must beat
- [x] Fit calibration temperature on valid only: temperature 0.5981 cuts typed-decisions test ECE 0.0938 -> 0.0204, beating the published best calibrated single (0.0279). Receipt in `runs/`
- [ ] Encode many questions against one shared state in a single pass instead of re-encoding the packed state per request
- [ ] Fit calibration temperature inside `train.py` so every new checkpoint ships calibrated instead of needing a separate pass afterwards

## Reliability

- [x] Quantify the OOD behavior: student-r2b shows mild conservatism, not overconfidence. 0 of 800 WANLI and 0 of 6 garbage-input answers reach p >= 0.9; emotion 14 of 800 confident, 1 confidently wrong. Receipt in `runs/`
- [ ] Chase down the residual confident errors on in-distribution-adjacent inputs (1 of 800 emotion answers confidently wrong)

## Internationalization

- [ ] Non-English checkpoints. The interface does not care about language, the weights do
