# OEV roadmap

Living list. Done items move to the changelog; every landed result is in [BENCHMARKS.md](BENCHMARKS.md).

## Next

- [ ] rebuild the MNLI specialist (~50 min GPU): trained once, hit WANLI `0.5645`, then lost to a runtime disconnect before download - the recipe is proven
- [ ] round 3 distillation: 6 teachers (typed, generalist, 3x b77, MNLI), 5 domains including NLI, pure-KL loss
- [ ] 4-member Banking77 ensemble (soup + re-tune + a + b): the live shot past `0.8584` toward Jev's `0.870`
- [ ] round 4 distillation: per-domain specialist teachers with balanced data, to close the generalist spread (typed `0.6480` vs specialist `0.7705`)
- [ ] coherence target: the distilled student scores `3/5` on the five decision-coherence scenarios (teacher baseline `4/5`) - add a conservative-decision term to distillation
- [ ] ANLI R1 specialist: the train split exists; a fine-tuned row first, adversarial transfer stays documented at chance

## Later

- [ ] INT8 / ONNX CPU deployment: export and quantization scripts are shipped in `scripts/`; fp32 p50 is `447 ms` on 8 threads and the INT8 bench is pending
- [ ] fold valid-split temperature fitting into `train.py` so new checkpoints ship calibrated
- [ ] multi-question shared-state encoding: one forward pass, many questions
- [ ] robustness: reduce mild overconfidence on out-of-distribution and garbage inputs
- [ ] more domain specialists (guardrails, moderation, RAG filtering)
- [ ] non-English checkpoints: the interface is language-agnostic, the weights are English-only
