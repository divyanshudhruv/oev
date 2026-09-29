# OEV roadmap

Measured results go to [BENCHMARKS.md](BENCHMARKS.md). Done items are checked with the number they landed.

## Near term

- [ ] Round 3 distillation with 6 teachers and 5 domains, MNLI included. Pure-KL loss only, because round 2 already showed what gold-CE does to the student
- [ ] Try the 4-member Banking77 ensemble. Soup plus re-tune plus both seeds is the most realistic shot at getting past `0.8584` toward Jev's `0.870`
- [ ] Rebuild the MNLI specialist. It trained once, hit WANLI `0.5645`, and the Colab artifact was lost before download. The recipe works, it needs about 50 GPU minutes. (The Hub file `mnli-oev-tiny.pt` currently holds multi-task weights from an earlier upload mistake - do not use it for NLI)

## Model quality

- [ ] Round 4 distillation: per-domain specialist teachers with balanced data. The point is depth. Right now the generalist scores `0.6480` on typed where the specialist hits `0.7705`. Training data to add, from the live head-to-head findings:
  - [ ] invoices and billing words appearing in technical contexts (the model hears "invoice" and ignores the 500 error)
  - [ ] multi-issue tickets with a "both" option, so two problems in one message stop collapsing to the first noun
  - [ ] benign security logins next to real incidents, to kill the alert-everything lean
  - [ ] sarcastic praise labeled very negative
  - [ ] trace pairs where the only difference is runtime, one alerting and one silent, so latency stops being invisible
  - [ ] a head-consistency term so "approve" never pairs with "clear mismatch"
- [ ] Bring coherence back up. Distillation traded decision conservatism for breadth, so the student scores `3/5` on the five decision-coherence scenarios where the teacher scores `4/5`. A conservative-decision term in the loss should close that
- [ ] Fine-tune on ANLI R1's train split. WANLI transfer worked, adversarial splits are a different animal, and a fine-tuned row would tell us how far it goes
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
