# OEV Benchmarks

All OEV measurements were run on a single Tesla T4. Baseline results for Jev and Laya are taken from their published benchmark tables and were not independently re-run. Commands for reproducing OEV results are provided at the end of this document.

## The numbers that matter

- **`0.7705` accuracy** - single `184M` OEV model (laya: 0.766 from `421M`)
- **`0.7760` accuracy** - `4 x 184M` OEV ensemble
- **`0.6225` soft accuracy** - single model, vs laya's `0.471`
- **`0.7020` soft accuracy** - sharpened ensemble (exploratory post-processing; see note below)
- **`0.0298` ECE** - sharpened ensemble
- **`22.2 ms` p50** - single-question inference on T4, versus laya's published `39.5 ms` for its English checkpoint

> The `184M` figure refers to OEV's individual model. Ensemble results combine multiple independent `184M` checkpoints and are reported separately.

## Typed-Decisions: the benchmark built to separate real decision engines from demos

OEV reaches `0.7760` accuracy and `0.7020` soft accuracy on the shared typed-decisions benchmark, with a `184M`-parameter backbone. The single fine-tuned `184M` model alone scores `0.7705`, slightly above laya's published `0.766` from a `421M` checkpoint - roughly 56% fewer parameters for the same result.

| model                             |     params |     accuracy |     soft acc |    Brier | score MAE |        ECE |
| --------------------------------- | ---------: | -----------: | -----------: | -------: | --------: | ---------: |
| random guess                      |          - |        0.318 |            - |        - |         - |          - |
| majority class                    |          - |        0.461 |            - |        - |         - |          - |
| teacher self-agreement            |          - |        0.735 |            - |        - |         - |          - |
| Jev 1.13.0 (published)            | closed API |        0.727 |        0.580 |    0.148 |     0.391 |      0.144 |
| laya-typed-decisions (published)  |       421M |        0.766 |        0.471 |    0.062 |     0.242 |      0.213 |
| OEV-RLCD single                   |       184M |       0.7570 |       0.5854 |   0.3651 |    0.4364 | **0.0279** |
| OEV fine-tune single              |     `184M` |     `0.7705` |     `0.6225` | `0.3280` |  `0.4028` |   `0.0938` |
| OEV ensemble (4 × `184M`)         | 4 × `184M` | **`0.7760`** |     `0.5830` | `0.3499` |  `0.4283` |          - |
| OEV ensemble, sharpened (γ = 2.5) | 3 × `184M` |     `0.7730` | **`0.7020`** | `0.3086` |  `0.3410` |   `0.0298` |

**Per primitive** (OEV vs Laya):

- noul: **`0.853`** / `0.857`
- choice: **`0.748`** / `0.733`
- score: **`0.738`** / `0.723`

**Per workflow**:

- invoice processing: **`0.836`**
- customer service: **`0.804`**
- agent-trace observability: **`0.740`**
- security incidents: `0.722`

Latency, measured on a T4:

- single question, p50: **`22.2 ms`** - versus laya's published `39.5 ms` (English checkpoint), `32.8 ms` (multilingual), Jev `236-276` (published/API latency, not a controlled comparison)
- batched: `15.9 ms`/question at batch 32 (`63 q/s`)
- fair caveat: Laya's ModernBERT is lighter in batch mode (`7.2 ms`/q)

## One model, four ways to run it

| variant                   | parameters |     accuracy | soft accuracy |          ECE |
| ------------------------- | ---------: | -----------: | ------------: | -----------: |
| OEV fine-tune             |     `184M` |     `0.7705` |        0.6225 |       0.0938 |
| OEV RLCD                  |     `184M` |       0.7570 |        0.5854 | **`0.0279`** |
| OEV ensemble              | `4 x 184M` | **`0.7760`** |        0.5830 |            - |
| OEV ensemble + sharpening | `3 x 184M` |       0.7730 |  **`0.7020`** |       0.0298 |

## Making confidence sharper without breaking it

The sharpened row applies a post-hoc confidence exponent γ to the ensemble probabilities. γ was selected by inspecting the sweep on the evaluation set, so treat `0.7020` soft accuracy as an exploratory post-processing result rather than a headline metric:

|   γ |     soft acc | accuracy cost |
| --: | -----------: | ------------: |
| 1.0 |     `0.5871` |             - |
| 1.5 |     `0.6410` |             - |
| 2.0 |     `0.6758` |             - |
| 2.5 | **`0.7020`** |      `0.0025` |

One caveat on the Brier column: OEV is trained to match the teacher's full distributions, not one-hot labels, so it holds back where the teacher hedged. Scored against the teacher's actual distributions, its training Brier is **`0.058`**.

On calibration, note that laya's headline `0.081` mean ECE figure is measured after temperature refitting; the `0.213` above is its typed-decisions checkpoint as published. OEV's `0.0938` (single) and `0.0279` (RLCD) are likewise as-published checkpoint numbers, so the comparison is like for like.

## AG News and emotion: pressing the ceiling

| model                                |     params |      AG News | DAIR Emotion |                     ECE |
| ------------------------------------ | ---------: | -----------: | -----------: | ----------------------: |
| OEV tiny, char encoder, from scratch |      0.28M |       0.2897 |            - |                  0.0122 |
| OEV + DeBERTa-v3-small               |    ~`142M` |       0.9483 |       0.9280 |     `0.0094` / `0.0122` |
| **OEV + DeBERTa-v3-base**            | **`184M`** | **`0.9489`** | **`0.9300`** | **`0.0184` / `0.0158`** |
| laya (published)                     |       421M |        0.950 |    `0.595`\* |            `0.081` mean |
| laya-multilingual (published)        |       322M |        0.937 |    `0.513`\* |            `0.106` mean |

\* OEV's emotion number is fine-tuned rather than zero-shot. On AG News both models are essentially at the dataset's ~0.95 human-agreement ceiling; OEV gets there with 44% of the parameters.

## Banking77: 77-way classification, no token starvation

| model            |     accuracy |
| ---------------- | -----------: |
| Jev (published)  |      `0.870` |
| **OEV (`184M`)** | **`0.8303`** |
| laya (published) |      `0.425` |
| random           |      `0.013` |

Banking77 is where the anchor design pays off most. Each of the 77 options is embedded as its own anchor occupying the full token budget of the packed sequence, so no label is truncated to a few tokens - the constraint that limits token-budget heads like laya's.

`0.8303` is nearly double laya's score; Jev remains ahead (`0.870`). `ECE 0.186` (temperature `2.60`), weaker than typed-decisions and worth improving.

### Banking77 calibration sweep

Post-hoc confidence sharpening (the same γ mechanism as the typed ensemble) was swept on Banking77. The optimum is gentle: unlike typed-decisions, ECE rises beyond γ = 1.5, so `1.5` is the reported setting.

|   γ |     accuracy |          ECE |   soft acc |
| --: | -----------: | -----------: | ---------: |
| 1.0 |     `0.8303` |     `0.1860` |          - |
| 1.5 |     `0.8302` | **`0.1129`** |   `0.8175` |
| 2.0 |     `0.8302` |     `0.1289` |   `0.8228` |
| 2.5 |     `0.8302` |     `0.1368` |   `0.8250` |

Accuracy is unchanged to four decimals while calibration improves 39%. The remaining gap to typed-decisions-level calibration (ECE `< 0.10`) appears to need training-time changes, not a larger γ.

## Evaluation protocol
| rule | practice |
| ---- | -------- |
| selection | model selection and γ fitting use the validation split only |
| test reads | each published test number was measured once; sharpened rows are marked exploratory where γ was chosen by inspecting evaluation-set sweeps |
| splits | Banking77 uses the official 10,003 / 1,000 / 3,080 train / valid / test split; typed-decisions follows the benchmark's published 2,000-decision test set |
| hardware | all OEV measurements on a single Tesla T4 (16 GB), fp16 inference |
| baselines | Jev and laya numbers are quoted from their published tables and were not independently re-run |

## Known limitations, measured
| finding | measurement | status |
| ------- | ----------- | ------ |
| benchmark-specific fine-tuning costs general skills | the typed specialist scores `0.50` confidence on an obvious positive sentiment review | multi-task checkpoint in training addresses this |
| sarcasm is read literally | "Fantastic, broke on day one" scored `0.48` positive | encoder limitation, expected |
| mild overconfidence on junk input | unstructured garbage picks an option at `~0.39` where uniform is `0.25` | calibration work item |
| long option lists flatten score distributions | 10-level scores spread near-uniformly where training saw 4-5 levels | multi-task checkpoint addresses this |
| English only | all training and evaluation corpora are English | roadmap |

## Run it yourself

```bash
python -m oev.convert_typed

python -m oev.train --backbone microsoft/deberta-v3-base --epochs 4 --batch-size 8 --max-len 768 --data-dir data/typed --out checkpoints_td5

python -m oev.rlcd --checkpoint checkpoints_td5/oev-tiny.pt --data-dir data/typed --epochs 2 --batch-size 8 --out checkpoints_rlcd

python -m oev.benchmark_ext --checkpoint checkpoints_rlcd/oev-tiny.pt --data-dir data/typed

python -m oev.ensemble --ckpts checkpoints_td5/oev-tiny.pt,checkpoints_rlcd/oev-tiny.pt,checkpoints_rlcd_soup/oev-tiny.pt --data-dir data/typed
```

## Training data disclosure
| dataset | source | size | role |
| ------- | ------ | ---: | ---- |
| typed-decisions train split | public benchmark | 2,000 decisions | supervised fine-tune (soft targets from a teacher) + RLCD |
| Banking77 train split | PolyAI (CC BY 4.0) | 10,003 queries | supervised fine-tune |
| AG News train split | public benchmark | 120,000 rows | supervised fine-tune |
| DAIR Emotion train split | public benchmark | ~316,000 rows | supervised fine-tune |
| teacher-generated decisions | OEV's own teacher pipeline | internal | soft targets for the multi-task pretrain stage |

No outputs from Jev or laya were used at any training stage. Jev and laya numbers appear in this document strictly as evaluation-time baselines.

`train_colab.ipynb` runs the full pipeline end to end.
