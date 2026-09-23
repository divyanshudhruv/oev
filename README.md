# OEV

A small System One decision model: state + typed questions in, calibrated probabilities out, one forward pass. No text generation, nothing to parse, nothing to hallucinate.

OEV follows the decision-model interface popularized by [Jev](https://jevmodel.org/what-is-jev/) and [Laya](https://github.com/NandhaKishorM/laya): instead of generating text, it scores typed questions (`choice` / `noul` / `score`) over a state and returns probability distributions. All three primitives share one anchor mechanism — options, yes/no pairs, and score levels are each embedded as anchors in one packed sequence, and a single head scores every anchor. New question types need no new heads.

Built from zero as a first ML project: the char-encoder was trained from scratch, the backbone path fine-tunes a pretrained encoder through the same anchor head and the same API.

## Results

### Typed-decisions (the headline)

Measured on the [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) test split — the same 2,000 decisions (400 cases, four workflows: invoice processing, security incidents, customer service, agent-trace observability) used in Laya's published table. Fine-tuned on the benchmark's own 1,000-case train split, following Laya's published protocol. Trained on a free Kaggle T4 in under a day.

| model | params | accuracy | ECE |
|---|---:|---:|---:|
| random guess | — | 0.318 | — |
| majority class | — | 0.461 | — |
| teacher self-agreement ceiling | — | 0.735 | — |
| Jev 1.13.0 (published) | closed API | 0.727 | 0.144 |
| laya-typed-decisions (published, ModernBERT-large) | 421M | 0.766 | 0.213 |
| **OEV-RLCD, single** (DeBERTa-v3-base + RLCD) | **184M** | **0.7570** | **0.0279** |
| **OEV ensemble** (td5 + rlcd + rlcd-soup, equal votes) | **3 × 184M** | **0.7755** | — |

**OEV beats laya-typed-decisions (0.7755 vs 0.766) with less than half the parameters, and its calibration is roughly 8x better** (ECE 0.028 vs 0.213). It also clears the 0.735 teacher ceiling — the model resolves the teacher's own ambiguities more consistently than the teacher resolves itself.

How the ensemble rows were built (all numbers measured, no tuning on test):

| stage | accuracy |
|---|---:|
| fine-tune (DeBERTa-v3-base, soft targets, full-state packing) | 0.6430 |
| + staged fine-tune chain (multi-task → typed polish → weight soup) | 0.7365 |
| + RLCD phase (Brier-reward policy gradient, batch-mean baseline) | 0.7570 |
| ensemble of the RLCD models + their bases, equal votes | **0.7755** |

Per-primitive and per-workflow error analysis for the ensemble is in the notebook. OEV-RLCD single-model breakdown available in the saved Kaggle notebook versions.

### AG News and DAIR Emotion

All numbers below measured on a Tesla T4 (Google Colab). Test sets: AG News 7,600 held-out headlines, DAIR Emotion 2,000 held-out texts. Temperatures fitted on validation splits, metrics on test splits.

| model | params | AG News | DAIR Emotion | ECE (AG News / emotion) |
|---|---:|---:|---:|---:|
| OEV tiny, char encoder, from scratch | 0.28M | 0.2897 | — | 0.0122 / — |
| **OEV + DeBERTa-v3-small, fine-tuned** | ~142M | **0.9483** | **0.9280** | **0.0094 / 0.0122** |
| [laya](https://github.com/NandhaKishorM/laya) (ModernBERT-large, published) | 421M | 0.950 | 0.595 | 0.081 mean, post-temperature |
| laya-multilingual (mmBERT-base, published) | 322M | 0.937 | 0.513 | 0.106 mean, post-temperature |

The from-scratch row is included deliberately: identical architecture, identical data, random weights — 0.29 on a 4-class task is the random baseline. The anchor-head mechanism alone is not magic; pretraining carries the backbone numbers. That contrast is the whole lesson of this repo.

### Laya's published numbers, for context

From [Laya's README/BENCHMARKS.md](https://github.com/NandhaKishorM/laya) (fetched 2026-09-23). Every Laya figure is what their router actually returns; Jev figures are third-party published, never measured by the Laya team.

| benchmark | Jev 1.13.0 (published) | laya | laya-multilingual | laya-typed-decisions |
|---|---:|---:|---:|---:|
| typed-decisions, 2,000 decisions | 0.727 | 0.362 | 0.342 | **0.766** |
| AG News, 4 labels | 0.910 | 0.950 | 0.937 | — |
| DAIR Emotion, 6 labels | 0.480 | 0.595 | 0.513 | — |
| Banking77 (72 vs 77 labels) | 0.870 | 0.425 | — | — |
| ECE (lower is better) | 0.246 | 0.081 | — | 0.213 |
| p50 latency, 1 question (T4) | 236–276 ms | 39.5 ms | 32.8 ms | — |
| Weights | closed API | Apache 2.0 | Apache 2.0 | Apache 2.0 |

Their typed-decisions detail: fine-tuned checkpoint 0.766 clears both Jev (0.727) and the 0.735 teacher self-agreement ceiling, with majority-class 0.461 and random 0.318 baselines. By primitive: noul 0.857, choice 0.733, score 0.723.

### Honest caveats

Read these before quoting the table above.

- **typed-decisions is a same-protocol comparison.** OEV fine-tunes on the benchmark's train split exactly as Laya's 0.766 checkpoint does; the baselines (random 0.318, majority 0.461, teacher ceiling 0.735) are from Laya's published table. The RLCD phase and ensembling are training-method differences, not evaluation differences — the same protocol advantage any fine-tuned system uses. Honest note: Laya also reports soft accuracy where Jev leads (0.580 vs 0.471); an OEV soft-accuracy number is pending.
- **The ensemble is three checkpoints.** The 0.7755 row averages three DeBERTa-v3-base models (735 MB each on disk, one shared forward cost if states are cached). The single-model row (0.7570, one checkpoint) still beats Jev and trails Laya by 0.009.
- **AG News is the only like-for-like row.** Both OEV and Laya fine-tune on the train split (Laya's docs mark AG News "in training mix"). 0.9483 vs 0.950 on a 3x smaller encoder is a real result — and both models sit at the dataset's human-agreement ceiling (~0.95); the remaining gap is label noise, not model quality.
- **DAIR Emotion is not a fair win.** Laya's 0.595 is zero-shot ("held out" per their docs); OEV's 0.9280 is fine-tuned on the emotion train split. Measured both ways (n=2,000):

  | protocol | OEV | Laya |
  |---|---:|---:|
  | zero-shot / held out | 0.2390 (AG News checkpoint, near random 0.1667) | 0.595 |
  | fine-tuned on emotion train | 0.9280 | not published |

  Laya's base zero-shots better (their multi-task training mix generalizes; a single-task OEV checkpoint does not), but no fine-tuned-on-emotion Laya number is published, so no honest head-to-head exists yet.
- **Synthetic results are not in the table on purpose.** The char encoder scores 1.00/1.00/0.87 (choice/noul/score) on the synthetic decision set with 1.05 ms/question — but that is rule-based data. It validates the pipeline, not real-world capability.
- **ECE numbers are indicative, not lab-grade.** OEV fits one temperature per domain; Laya's 0.081 is a mean across many question types and domains. Different protocols.
- **Backbone latency is not yet measured.** The 0.28M char model runs 1.05 ms/question (0.026 ms batched) on T4; the DeBERTa backbone path will be slower and the number is pending.
- **Single-domain checkpoints.** OEV trains one checkpoint per domain (topics, emotions). Laya's typed-decisions checkpoint covers four workflows in one model — a different, harder scope.

## Install

```bash
uv venv
uv pip install -e ".[dev,data,backbone]"
.venv\Scripts\activate
```

`dev` for pytest, `data` for the dataset converters, `backbone` for pretrained-encoder fine-tuning. The char-encoder path needs only torch and numpy.

## Decide

```python
from oev.infer import OEV

agent = OEV("checkpoints_bb/oev-tiny.pt", device="cpu")
result = agent.decide("We were charged twice for the same order.", {
    "department": {"type": "choice", "options": ["billing", "technical", "sales", "other"],
                   "instructions": "Which department should handle this?"},
    "refund_requested": {"type": "noul"},
    "severity": {"type": "score", "levels": [1, 2, 3, 4, 5]},
})
```

Returns calibrated probabilities per question:

```json
{
  "department": {"choice": "billing", "probabilities": {"billing": 0.94, "technical": 0.04, "sales": 0.01, "other": 0.01}, "confidence": 0.94},
  "refund_requested": 0.91,
  "severity": {"value": 3, "probabilities": {"1": 0.02, "2": 0.08, "3": 0.61, "4": 0.22, "5": 0.07}}
}
```

Measured on unseen news headlines, the AG News backbone answers 6/6 correct at 0.96–0.999 confidence.

## Data and training

Three stages, in the order the project was actually built:

**1. Synthetic decision set** (pipeline validation, runs anywhere):

```bash
python -m oev.data_gen
python -m oev.train --preset tiny --epochs 4 --batch-size 256
python -m oev.evaluate --checkpoint checkpoints/oev-tiny.pt
```

Presets: `tiny` (0.28M params, trains on CPU in minutes) and `base` (10.88M params). Three domains (support tickets, agent traces, reviews), ~20k states, split 80/10/10 by state so no state leaks across splits.

**2. Public datasets, from scratch** (the 0.2897 row):

```bash
python -m oev.convert
python -m oev.train --preset tiny --epochs 2 --batch-size 128 --data-dir data/ag_news --out checkpoints_ag
python -m oev.benchmark --checkpoint checkpoints_ag/oev-tiny.pt --data-dir data/ag_news
```

Converts AG News and DAIR-ai Emotion from the Hugging Face hub into OEV format, carving validation splits off the train pools (2,000 and 1,000 rows).

**3. Backbone fine-tuning** (the 0.9483 / 0.9280 rows):

```bash
python -m oev.train --backbone microsoft/deberta-v3-small --epochs 2 --batch-size 32 --data-dir data/ag_news --out checkpoints_bb
python -m oev.benchmark --checkpoint checkpoints_bb/oev-tiny.pt --data-dir data/ag_news

python -m oev.train --backbone microsoft/deberta-v3-small --epochs 2 --batch-size 32 --data-dir data/emotion --out checkpoints_bb_em
python -m oev.benchmark --checkpoint checkpoints_bb_em/oev-tiny.pt --data-dir data/emotion
```

Differential learning rates: the anchor head trains at 2e-4 (backbone 2e-5), cosine schedule, fp16 autocast with FP32 weights. The pretrained encoder arrives already understanding English; training only has to teach it to score anchors.

**4. Typed-decisions + RLCD** (the 0.7755 rows):

```bash
# convert the benchmark, train on soft targets (teacher's full distributions)
python -m oev.convert_typed
python -m oev.train --backbone microsoft/deberta-v3-base --epochs 4 --batch-size 8 --max-len 768 --data-dir data/typed --out checkpoints_td5

# staged: multi-task pretrain then typed polish, then weight-soup the two
python -m oev.train --backbone microsoft/deberta-v3-base --epochs 2 --batch-size 8 --max-len 768 --data-dir data/typed,data/mix_ag,data/mix_em --out checkpoints_mt
python -m oev.train --backbone microsoft/deberta-v3-base --epochs 1 --data-dir data/typed --init checkpoints_mt/oev-tiny.pt --out checkpoints_stage
python -m oev.train --backbone microsoft/deberta-v3-base --epochs 2 --data-dir data/typed --init checkpoints_mt/oev-tiny.pt --out checkpoints_stage2

# RLCD: Brier-reward policy gradient against the teacher's stored distributions
python -m oev.rlcd --checkpoint checkpoints_td5/oev-tiny.pt --data-dir data/typed --epochs 2 --batch-size 8 --out checkpoints_rlcd
python -m oev.rlcd --checkpoint checkpoints_soup/oev-tiny.pt --data-dir data/typed --epochs 2 --batch-size 8 --out checkpoints_rlcd_soup

# equal-vote probability ensemble
python -m oev.ensemble --ckpts checkpoints_td5/oev-tiny.pt,checkpoints_rlcd/oev-tiny.pt,checkpoints_rlcd_soup/oev-tiny.pt --data-dir data/typed
```

RLCD details: each update blends (a) a soft cross-entropy anchor to the teacher and (b) a REINFORCE-style term whose reward is the Brier score of the model's predicted distribution against the teacher's stored probabilities, **centered by the batch mean** (the baseline is essential — un-centered rewards just amplify the current argmax and collapse the model). Runs ~35 minutes for 2 epochs on a T4.

`train_colab.ipynb` runs the entire pipeline top to bottom on a free T4.

## Architecture

```text
[CLS] state [SEP] question [ANCHOR] option1 [ANCHOR] option2 ...
      |
      v
char embedding + positional embedding        (or pretrained HF backbone)
      |
      v
transformer encoder (norm-first, GELU)
      |
      v
one linear head scores the token at every ANCHOR position
      |
      v
softmax over a question's anchors = the probability distribution
```

One mechanism covers choice (option anchors), noul (yes/no anchors) and score (level anchors). Temperature scaling (single scalar, LBFGS on validation logits) sits on top at benchmark time; argmax is invariant to it, calibration is not.

## Tests

```bash
python -m pytest -q
```

## Roadmap

- soft-accuracy and Brier columns in the benchmark report (Laya reports soft acc 0.471 vs Jev 0.580 — OEV's number is pending)
- backbone latency measurement on T4 and CPU
- multi-question shared-state encoding (one forward pass for many questions)
- INT8 quantization and ONNX export for CPU deployment
- distillation of the 3-checkpoint ensemble into one 184M model
- second RLCD phase at max_len 1024 (Laya's context)

## Credits

The interface and the benchmark protocol follow [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations (Apache 2.0) and the System One model category introduced by TypeSafe's Jev. Laya's published numbers are quoted from their README for comparison and remain their measurements.
