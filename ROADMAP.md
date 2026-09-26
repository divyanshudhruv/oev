# OEV roadmap

What I want to get done, roughly in order. Measured results go to [BENCHMARKS.md](BENCHMARKS.md).

## Next

- [ ] Rebuild the MNLI specialist. It trained once, hit WANLI `0.5645`, and then the Colab session died before I could download the file. The recipe works, it just needs about 50 GPU minutes
- [ ] Round 3 distillation with 6 teachers and 5 domains, MNLI included. Pure-KL loss only, because round 2 already showed what gold-CE does to the student
- [ ] Try the 4-member Banking77 ensemble. Soup plus re-tune plus both seeds is the most realistic shot at getting past `0.8584` toward Jev's `0.870`
- [ ] Round 4 distillation: per-domain specialist teachers with balanced data. The point is depth. Right now the generalist scores `0.6480` on typed where the specialist hits `0.7705`
- [ ] Bring coherence back up. Distillation traded decision conservatism for breadth, so the student scores `3/5` on the five decision-coherence scenarios where the teacher scores `4/5`. A conservative-decision term in the loss should close that
- [ ] Fine-tune on ANLI R1's train split. WANLI transfer worked, adversarial splits are a different animal, and a fine-tuned row would tell us how far it goes

## Later

- [x] Run the INT8 ONNX bench. It landed at `54.2 ms` p50 on 8 threads, about 4.7x under the torch fp32 figure on the same machine, and the artifact is `228 MB` instead of `735 MB`
- [ ] Try fp16 autocast on the eval path. Everything currently runs fp32, and autocast should roughly double GPU throughput if accuracy holds
- [ ] Fit calibration temperature inside `train.py` so every new checkpoint ships calibrated instead of needing a separate pass afterwards
- [ ] Encode many questions against one shared state in a single pass instead of re-encoding the packed state per request
- [ ] Chase down the mild overconfidence the models still show on out-of-distribution and garbage inputs
- [ ] Train more specialists for guardrails, moderation and RAG filtering
- [ ] Non-English checkpoints. The interface does not care about language, the weights do
