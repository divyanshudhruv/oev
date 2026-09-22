# OEV

A small decision model: state + typed questions (choice / noul / score) in, calibrated probabilities out. No text generation.

OEV follows the System One model interface popularized by Jev and Laya: instead of generating text, it scores typed questions over a state in a single forward pass and returns calibrated probability distributions. All three question types share one anchor mechanism: options, yes/no, and score levels are each embedded as anchors in one packed sequence, and a single head scores every anchor.

## Install

```bash
uv venv
uv pip install -e ".[dev]"
.venv\Scripts\activate
```

## Data

Synthetic generator covering three domains (support tickets, agent traces, reviews), ~20k states producing ~60k state+question examples, split 80/10/10 by state so no state appears in two splits.

```bash
python -m oev.data_gen
```

## Train

```bash
python -m oev.train --preset tiny --epochs 4 --batch-size 256
```

Presets: `tiny` (0.28M params, trains on CPU in minutes) and `base` (10.88M params, prefers a GPU). Every epoch evaluates on the validation split and the best checkpoint is kept in `checkpoints/oev-<preset>.pt`.

## Evaluate

```bash
python -m oev.evaluate --checkpoint checkpoints/oev-tiny.pt
```

Measured on a Tesla T4 (Google Colab), 6,000 held-out test questions, 4 epochs (tiny) / 1 epoch (base):

| metric | tiny | base |
|---|---|---|
| accuracy choice | 1.0000 | 1.0000 |
| baseline choice | 0.2075 | 0.2075 |
| accuracy noul | 1.0000 | 1.0000 |
| baseline noul | 0.7031 | 0.7031 |
| accuracy score | 0.8705 | 0.8635 |
| baseline score | 0.3440 | 0.3440 |
| ECE | 0.0017 | 0.0032 |
| params | 282,625 | 10,882,561 |
| disk MB | 1.14 | 43.56 |
| latency ms / question | 1.05 | 2.25 |
| latency ms / question batched | 0.026 | 0.038 |

Baselines are per-question majority class. The tiny preset matches or beats base on every metric at 1/38th the size: for this decision task, data signal saturates well below 1M parameters. Note the accuracies reflect a synthetic rule-based dataset; real-world data (e.g. Open-Jev) is the next milestone.

## Decide

```python
from oev.infer import OEV

agent = OEV("checkpoints/oev-tiny.pt", device="cpu")
result = agent.decide("We were charged twice for the same order.", {
    "department": {"type": "choice", "options": ["billing", "technical", "sales", "other"]},
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

## Architecture

```text
[CLS] state [SEP] question [ANCHOR] option1 [ANCHOR] option2 ...
      │
      ▼
char embedding + positional embedding
      │
      ▼
transformer encoder (2 or 6 layers, norm-first, GELU)
      │
      ▼
one linear head scores the token at every ANCHOR position
      │
      ▼
softmax over a question's anchors = the probability distribution
```

One mechanism covers choice (option anchors), noul (yes/no anchors) and score (level anchors). New question types need no new heads.

## Tests

```bash
python -m pytest -q
```

## Roadmap

- harder synthetic data: mixed-domain states, ambiguity, lexical variety
- Open-Jev dataset conversion (520k real-ish Jev-format examples)
- multi-question shared-state encoding (one forward pass for many questions)
- distillation from a larger teacher (Apache-2.0 Laya checkpoints are eligible)
- INT8 quantization and ONNX export for CPU deployment
