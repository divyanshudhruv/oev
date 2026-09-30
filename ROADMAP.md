# OEV roadmap

Measured results live in [BENCHMARKS.md](BENCHMARKS.md); completed items are listed there with their numbers.

## Shipped

- Round 3 distillation: 3 teachers, 5 domains including MNLI, pure-KL loss. Checkpoint on the Hub (`student-r3-oev-tiny.pt`), superseded by round 4
- Banking77 soup4: project-best single-file Banking77; ensemble saturation at the 5th member, so pushing past Jev's `0.870` needs a different lever
- MNLI specialist rebuilt; the Hub file `mnli-oev-tiny.pt` holds the real specialist
- Round 4 distillation: 10 domains with the live head-to-head fix sets mixed in, warm-started from round 3. Checkpoint on the Hub (`student-r4-oev-tiny.pt`)
- ANLI R1 fine-tune of the rebuilt specialist: first adversarial NLI row, WANLI transfer improved over its own init
- INT8 ONNX CPU deployment, fp16 eval, batched-latency bound for shared-state encoding, calibration temperature fitted on valid only

## Model quality

- Push Banking77 past Jev's `0.870`: current best `0.8594`. The lever is lineage diversity, data augmentation or distillation, not more warm-starts
- Close the coherence gap: distillation traded decision conservatism for breadth (student `3/5` vs teacher `4/5` on the five coherence scenarios). A conservative-decision term in the loss should close it
- Calibrate the NLI line: the ANLI checkpoint is overconfident out of domain (WANLI ECE `0.2308`)
- Train more specialists for guardrails, moderation and RAG filtering

## Inference

- Encode many questions against one shared state in a single pass instead of re-encoding the packed state per request
- Fit calibration temperature inside `train.py` so every new checkpoint ships calibrated
- OEV as the action scorer inside agentic loops: a generative agent proposes candidate actions, OEV scores which fits the state in one 22 ms pass

## Reliability

- Chase down residual confident errors on in-distribution-adjacent inputs

## Internationalization

- Non-English checkpoints. The interface does not care about language, the weights do
