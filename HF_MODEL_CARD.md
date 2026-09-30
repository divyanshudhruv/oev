---
license: apache-2.0
language:
  - en
tags:
  - decision-model
  - classification
  - calibration
  - system-one
  - deberta
---

<p align="center">
  <img src="https://raw.githubusercontent.com/divyanshudhruv/oev/refs/heads/main/assets/banner_typed.png" alt="OEV-typed banner" width="100%" />
</p>

A `184M`-parameter decision engine. State + typed questions in, calibrated probability distributions out, one forward pass, `22.2 ms` p50 on T4.

OEV reads a state, scores typed questions over it in one forward pass, and returns calibrated probability distributions. There is no text generation step, so there is nothing to parse and nothing to hallucinate.

It answers three kinds of questions over any text state: `choice` (pick a label), `noul` (yes/no), and `score` (ordinal levels). All options are packed into one sequence as anchors, and a single head scores them all at once.

## Results (typed-decisions benchmark, 2,000 decisions)

Fine-tuned on the benchmark's train split, same protocol as [laya-typed-decisions](https://github.com/NandhaKishorM/laya). OEV numbers were measured on a single Tesla T4; Jev and laya values are quoted from their published tables.

| model                                                |     params |     accuracy |     soft acc |          ECE | p50 latency (T4) |
| ---------------------------------------------------- | ---------: | -----------: | -----------: | -----------: | ---------------: |
| Jev 1.13.0 (published)                               | closed API |        0.727 |        0.580 |        0.144 |     236-`276 ms` |
| laya-typed-decisions (published)                     |       421M |        0.766 |        0.471 |        0.213 |   32.8-`39.5 ms` |
| **OEV single (oev-base-td5)**                        | **`184M`** | **`0.7705`** |       0.6225 |       0.0938 |    **`22.2 ms`** |
| **OEV calibrated single (oev-base-rlcd-soup)**       | **`184M`** |       0.7570 |       0.5854 | **`0.0279`** |    **`22.2 ms`** |
| **OEV temperature-calibrated (oev-base-td5, T=0.598)**               | **`184M`** |       0.7705 |            - | **`0.0204`** |    **`22.2 ms`** |

The temperature-calibrated row is the best calibrated single: temperature `0.598` fitted on the valid split only, accuracy unchanged by monotone scaling. Use it via `OEV(checkpoint, temperature=0.598)`.
| **OEV ensemble (all four checkpoints, equal votes)** | 4 × `184M` | **`0.7760`** |       0.5830 |            - |                - |

Per workflow (ensemble): invoice `0.836`, customer service `0.804`, agent-trace `0.740`, security incidents `0.722` (OEV leads three of the four workflows). Per primitive: noul `0.853`, choice `0.748`, score `0.738`.

## Checkpoints

| file                        | what it is                                                                                                                                                                                                                     |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **student-r3-oev-tiny.pt** | **distilled generalist - recommended default.** Round-3 pure-KL distill, one model for everything: typed `0.6895`, Banking77 `0.8370`, emotion zero-shot `0.8650` (beats laya `0.595` by +27) |
| student-r4-oev-tiny.pt      | round-4 generalist, warm-started from r3 with 5 head-to-head fix domains mixed in: typed `0.6985`, emotion zero-shot `0.8700`, Banking77 `0.8188` (stated trade for the fixes). Distill underconfidence: apply gamma sharpening for calibrated probabilities |
| student-r2b-oev-tiny.pt     | previous distilled generalist (round 2b): typed `0.6480`, Banking77 `0.7964`, emotion zero-shot `0.6505`. Superseded by the round-3 student |
| oev-base-td5.pt             | typed-decisions specialist (0.7705; with temperature 0.598 it is the most calibrated single at ECE 0.0204) - best when you only need agent-decision scoring                                                                    |
| oev-base-rlcd-soup.pt       | most calibrated single (ECE 0.0279). Prefer this one when the probabilities feed automated decisions.                                                                                                                          |
| oev-base-rlcd.pt            | RLCD fine-tune (Brier-reward policy gradient against teacher distributions)                                                                                                                                                    |
| oev-base-rlcd-seed1.pt      | second RLCD seed - for ensembling                                                                                                                                                                                              |
| b77-oev-tiny.pt             | Banking77 specialist (77-way intents) - see the Banking77 section below                                                                                                                                                        |
| b77a-oev-tiny.pt            | Banking77 warm-start re-tune (0.8403 historical value) - for the b77 ensemble                                                                                                                                                  |
| b77b-oev-tiny.pt            | Banking77 warm-start re-tune, second ensemble member                                                                                                                                                                           |
| b77soup-oev-tiny.pt         | weight-average of the three b77 members (0.8584)                                                                                                                                                                              |
| b77soup4-oev-tiny.pt        | 4-member weight soup with a soup re-tune member - best single-file b77 accuracy (`0.8594`)                                                                                                                                     |
| mnli-oev-tiny.pt            | MNLI specialist, WANLI zero-shot `0.5265`. Warm-start for ANLI fine-tunes and NLI-domain distillation     |
| anli-r1-oev-tiny.pt         | ANLI R1 fine-tune of the MNLI specialist: `0.5750` in-domain (chance `0.333`), WANLI zero-shot `0.5690` - the project's best NLI transfer. Overconfident OOD (ECE `0.2308`); calibrate before automated gating     |

For the ensemble results, average the softmax probabilities of the members with equal weights: typed-decisions `0.7760` (four files), Banking77 `0.8529` with ECE `0.0595` (three files: `b77-oev-tiny.pt` + `b77a-oev-tiny.pt` + `b77b-oev-tiny.pt`).

## Usage

```python
import torch, torch.nn.functional as F
from oev.evaluate import load_model
from oev.tokenizer_hf import HFTokenPacker
from huggingface_hub import hf_hub_download

path = hf_hub_download("divyanshudhruv/oev-typed", "student-r3-oev-tiny.pt")
model = load_model(path, "cuda")
packer = HFTokenPacker(model.cfg["backbone"])

state = "We were charged twice for the same order."
question = {"name": "department", "type": "choice",
            "instructions": "Which department should handle this?",
            "options": ["billing", "technical", "other"], "answer": "billing"}

# The Jev/TypeSafe `criteria` schema (used by laya and Kev) is also accepted:
# {"name": "department", "type": "choice", "instructions": "...",
#  "criteria": {"billing": "invoices, payments", "technical": "bugs, outages"}}
# descriptions are ignored by OEV; the option names are what it scores.

ids, anchors, _ = packer.pack(state, question, model.cfg["max_len"])
with torch.no_grad():
    logits = model(torch.tensor([ids]).cuda(),
                   torch.zeros(1, len(ids), dtype=torch.bool).cuda(),
                   torch.tensor([anchors]).cuda())
probs = F.softmax(logits[0].float(), dim=-1)
# probs ≈ billing 0.94, technical 0.04, other 0.02
```

Try the [interactive Space demo](https://huggingface.co/spaces/divyanshudhruv/oev-demo), or install the package:

```bash
pip install oev
```

Training and evaluation code, the ensemble command, dataset converters and tests are all in the [GitHub repo](https://github.com/divyanshudhruv/oev).

## Training recipe

1. **Soft-target fine-tune** - typed-decisions train split, teacher's full distributions as targets, full-state packing, DeBERTa-v3-base backbone, differential LRs (head 2e-4 / backbone 2e-5), 4 epochs, batch 8, 768 tokens.
2. **Staged variants** - multi-task pretrain (AG News + emotion + typed), then typed polish, then weight soup.
3. **RLCD** - REINFORCE with a Brier-reward against the teacher's stored probabilities, batch-mean centered, blended with the imitation anchor (α = 0.5). Two seeds + one soup-init variant.
4. **Ensemble** - equal-weight probability averaging across the four checkpoints.

The whole pipeline trains in about a day on one T4. `train_colab.ipynb` runs it end to end.

## Notes

- Fine-tuned on each benchmark's own train split; comparison numbers are from the respective published tables.
- OEV trains to match the teacher's full distributions (training Brier vs teacher: `0.058`).
- English models. OEV latency is p50 of 50 warmed runs including sync on T4.

## License

Apache 2.0. The interface and benchmark protocol follow [Laya](https://github.com/NandhaKishorM/laya) and the decision-model category introduced by TypeSafe's Jev; their published numbers are quoted for comparison and remain their measurements.

## Banking77

Fine-tuned checkpoints: `b77-oev-tiny.pt` plus two warm-started re-tunes (`b77a-oev-tiny.pt`, `b77b-oev-tiny.pt`). Each of the 77 intents is embedded as its own anchor occupying the full token budget, so no label is truncated.

> [!WARNING]
> The published Jev Banking77 figure uses 72 labels, while OEV uses 77. The values are not a controlled head-to-head comparison.

| model                                                 | params   |     accuracy |          ECE |
| ----------------------------------------------------- | -------- | -----------: | -----------: |
| Jev (published)                                       | closed   |        0.870 |            - |
| **OEV weight soup, one file (`b77soup4-oev-tiny.pt`)** | **184M** | **`0.8594`** |     `0.1062` |
| OEV weight soup (`b77soup-oev-tiny.pt`)               |    184M |     `0.8584` |     `0.0965` |
| OEV 4-member ensemble                                 | 4 x 184M |     `0.8568` | **`0.0583`** |
| OEV ensemble (3 checkpoints)                          | 3 x 184M |     `0.8529` |     `0.0595` |
| OEV historical warm-start re-tune                     | 184M     |       0.8403 |       0.1905 |
| OEV single (first release)                            | 184M     |       0.8303 |       0.1860 |
| laya                                                  | 421M     |        0.425 |            - |

The soup (weight-space average of the three members) beats the probability ensemble with a single 735MB artifact and reaches coverage `0.7604` at a `<=5%` error budget (`AURC 0.0389`), so 76% of decisions can be automated under the stated budget. On CPU the ONNX INT8 build runs at `54.2 ms` p50 on 8 threads (`228 MB` artifact); the release measurement with torch fp32 was `447 ms`. No competing decision model publishes any CPU latency.
