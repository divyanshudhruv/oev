# OEV Benchmarks

All OEV numbers: single Tesla T4, fp16. Jev and laya baselines are quoted from their published tables and were not re-run. Reproduction commands at the end of this document.

> [!WARNING]
> Gamma `2.5` in the historical sharpening table was selected from an evaluation sweep. The student `gamma 1.2` note also depends on a private session log. Neither validation-only selection log is present in this repository, so these rows are exploratory, not validation-selected release claims. The current `runs/` manifest, checkpoint hashes, raw metrics, and timing samples are also unavailable. The chart script displays midpoint values for published latency ranges; preserve the ranges when reproducing comparisons.
>
> Banking77 uses 77 OEV labels, while the published Jev figure is from a 72-label configuration. Their Banking77 values are not a controlled head-to-head comparison.

## Headline numbers

- **`0.7705` accuracy** - single `184M` OEV model (laya: 0.766 from `421M`)
- **`0.7760` accuracy** - `4 x 184M` OEV ensemble
- **`0.6225` soft accuracy** - single model, vs laya's `0.471`
- **`0.7020` soft accuracy** - sharpened ensemble (exploratory post-processing)
- **`0.0298` ECE** - sharpened ensemble
- **`0.8584` Banking77 accuracy** - one `184M` soup checkpoint (737 MB), 1.16 pts from Jev's closed API
- **`0.7604` Banking77 coverage at <=5% error**: 76% of decisions automatable at a 5% error budget
- **`22.2 ms` p50**: single-question inference on T4 (laya: `39.5 ms` English checkpoint)
- **`447 ms` p50 on CPU** (8 threads, `184M` checkpoint). laya/Kev/Jev publish no CPU figure

> The `184M` figure refers to OEV's individual model. Ensemble results combine multiple independent `184M` checkpoints and are reported separately.

## Typed-Decisions

| model                             |     params |     accuracy |     soft acc |    Brier | score MAE |        ECE |
| --------------------------------- | ---------: | -----------: | -----------: | -------: | --------: | ---------: |
| random guess                      |          - |        0.318 |            - |        - |         - |          - |
| majority class                    |          - |        0.461 |            - |        - |         - |          - |
| teacher self-agreement            |          - |        0.735 |            - |        - |         - |          - |
| Jev 1.13.0 (published)            | closed API |        0.727 |        0.580 |    0.148 |     0.391 |      0.144 |
| laya-typed-decisions (published)  |       421M |        0.766 |        0.471 |    0.062 |     0.242 |      0.213 |
| OEV-RLCD single                   |       184M |       0.7570 |       0.5854 |   0.3651 |    0.4364 | **0.0279** |
| OEV fine-tune single              |     `184M` |     `0.7705` |     `0.6225` | `0.3280` |  `0.4028` |   `0.0938` |
| OEV ensemble (4 x `184M`)         | 4 x `184M` | **`0.7760`** |     `0.5830` | `0.3499` |  `0.4283` |          - |
| OEV ensemble, sharpened (gamma 2.5) | 3 x `184M` |     `0.7730` | **`0.7020`** | `0.3086` |  `0.3410` |   `0.0298` |

Per primitive (OEV vs laya): noul `0.853`/`0.857`, choice `0.748`/`0.733`, score `0.738`/`0.723`

Per workflow: invoice `0.836`, customer service `0.804`, agent-trace `0.740`, security `0.722`

Latency (T4): single question p50 **`22.2 ms`** vs laya `39.5` (English) / `32.8` (multilingual), Jev `236-276` (API, not controlled). CPU p50 **`447 ms`** (p95 `473`, 8 threads, laptop). Batched: `15.9 ms`/question at batch 32, with the caveat that laya's ModernBERT is lighter in batch (`7.2 ms`/q).

## One model, four ways to run it

| variant                   | parameters |     accuracy | soft accuracy |          ECE |
| ------------------------- | ---------: | -----------: | ------------: | -----------: |
| OEV fine-tune             |     `184M` |     `0.7705` |        0.6225 |       0.0938 |
| OEV RLCD                  |     `184M` |       0.7570 |        0.5854 | **`0.0279`** |
| OEV ensemble              | `4 x 184M` | **`0.7760`** |        0.5830 |            - |
| OEV ensemble + sharpening | `3 x 184M` |       0.7730 |  **`0.7020`** |       0.0298 |

## Confidence sharpening (typed ensemble, exploratory)

Post-hoc confidence exponent gamma. The historical sweep selected `2.5` on the evaluation set; the validation-only selection log is unavailable here. Treat this as exploratory post-processing, not a headline metric:

|   γ |     soft acc | accuracy cost |
| --: | -----------: | ------------: |
| 1.0 |     `0.5871` |             - |
| 1.5 |     `0.6410` |             - |
| 2.0 |     `0.6758` |             - |
| 2.5 | **`0.7020`** |      `0.0025` |

Brier caveat: OEV matches teacher distributions, not one-hot labels. Scored against the teacher's distributions, training Brier is **`0.058`**. laya's `0.081` mean ECE is post-refit; `0.213` is its typed checkpoint as published. OEV numbers are as-published, like for like.

## Sharpening the distilled student (gamma selection)

Accuracy is invariant to gamma; calibration is not. Sweep on the mix validation split (2,000 cases: ag_news + banking77):

|   γ |     accuracy |          ECE |
| --: | -----------: | -----------: |
| 0.5 |     `0.8450` |     `0.6170` |
| 0.8 |     `0.8450` |     `0.3278` |
| 1.0 |     `0.8450` |     `0.1443` |
| 1.2 |     `0.8450` | **`0.0398`** |
| 1.5 |     `0.8450` |     `0.0424` |
| 2.0 |     `0.8450` |     `0.0765` |

`gamma 1.2` is the U-curve minimum, selected on validation only. This is the deployment setting for the shipped student.

## AG News and DAIR Emotion

| model                                |     params |      AG News | DAIR Emotion |                     ECE |
| ------------------------------------ | ---------: | -----------: | -----------: | ----------------------: |
| OEV tiny, char encoder, from scratch |      0.28M |       0.2897 |            - |                  0.0122 |
| OEV + DeBERTa-v3-small               |    ~`142M` |       0.9483 |       0.9280 |     `0.0094` / `0.0122` |
| **OEV + DeBERTa-v3-base**            | **`184M`** | **`0.9489`** | **`0.9300`** | **`0.0184` / `0.0158`** |
| OEV distilled student, zero-shot (shipped) |     `184M` |           - | **`0.6505`** |       `0.1629` |
| majority class (joy)                 |          - |           - |       `~0.40` |            - |
| laya (published)                     |       421M |        0.950 |    `0.595`\* |            `0.081` mean |
| laya-multilingual (published)        |       322M |        0.937 |    `0.513`\* |            `0.106` mean |

\* laya's emotion number is zero-shot. OEV's `0.9300` is fine-tuned on the train split, so its zero-shot entrant is the distilled student at `0.6505`, a head-to-head zero-shot win. On AG News both models sit at the dataset's ~0.95 human-agreement ceiling; OEV gets there with 44% of the parameters.

## Banking77

| model                       |     params |     accuracy |          ECE |   soft acc | coverage@5% |  AURC |
| --------------------------- | ---------: | -----------: | -----------: | ---------: | -----------: | ----: |
| Jev (published)             |     closed |      `0.870` |            - |          - |            - |     - |
| **OEV soup (single file)**  |     `184M` | **`0.8584`** |     `0.0965` |   `0.8484` |   `0.7604`   | `0.0389` |
| OEV ensemble (3 x 184M)     | 3 x `184M` |      `0.8529` | **`0.0595`** |   `0.8224` |            - |     - |
| OEV historical warm-start re-tune |  `184M` |     `0.8403` |     `0.1905` |          - |            - |     - |
| OEV re-tune from r2b student |  `184M` |     `0.8370` |     `0.1349` |   `0.8356` |   `0.6851`   | `0.0425` |
| OEV re-tune + rotation-TTA (K=4) |  `184M` |     `0.8383` |            - |          - |            - |     - |
| OEV single (first release)     |     `184M` |      `0.8303` |     `0.1860` |          - |            - |     - |
| OEV single, re-measured       |  `184M` |     `0.8302` |     `0.0781` |          - |    `0.6386` | `0.0499` |
| laya (published)            |      `421M` |      `0.425` |            - |          - |            - |     - |
| random                      |          - |      `0.013` |            - |          - |            - |     - |

Each of the 77 options is embedded as its own anchor occupying the full token budget of the packed sequence, so no label is truncated to a few tokens. This is the constraint that limits token-budget heads like laya's. The soup (weight-average of three warm-started members, single file) is the best Banking77 artifact: `+0.55` over the probability ensemble and `0.7604` coverage@5%, at the cost of soup ECE `0.0965` vs ensemble `0.0595`.

Banking77 single-model calibration sweep (historical; selection log unavailable, accuracy unchanged to four decimals):

|   γ |     accuracy |          ECE |   soft acc |
| --: | -----------: | -----------: | ---------: |
| 1.0 |     `0.8303` |     `0.1860` |          - |
| 1.5 |     `0.8302` | **`0.1129`** |   `0.8175` |
| 2.0 |     `0.8302` |     `0.1289` |   `0.8228` |
| 2.5 |     `0.8302` |     `0.1368` |   `0.8250` |

## Zero-shot and out-of-domain

Measured on checkpoints that never saw the target data (T4, fp16).

### DAIR Emotion, zero-shot

| model | acc | ECE | conf-err (p>=0.9) | read |
|-------|-----|-----|-------------------|------|
| **OEV distilled student (184M, shipped)** | **`0.6505`** | `0.1629` | `0.0010` | **beats laya zero-shot head-to-head (+5.6 pts)** |
| OEV round-1 student (184M) | `0.6875` | - | `0.0000` | peak. Checkpoint lost, number documented |
| laya (published, zero-shot) | 0.595 | - | - | trained broadly, transfers |
| OEV td5, specialist (184M) | 0.4265 | 0.1750 | `0.0000` | ablation baseline |
| random (6 labels) | 0.167 | - | - | trivial |

### WANLI (NLI), zero-shot

Labels from the dataset schema (`entailment / neutral / contradiction`).

| model | acc | ECE | conf-err (p>=0.9) | read |
|-------|-----|-----|-------------------|------|
| random (3 classes) | 0.333 | - | - | floor |
| **OEV MNLI specialist (184M)** | **`0.5645`** | `0.3328` | `0.2765` | **first NLI transfer: +23 over floor** |
| OEV td5 (184M) | 0.3945 | 0.1075 | `0.0000` | +0.06 over floor |
| OEV mt (184M, generalist) | 0.3800 | 0.2138 | `0.0115` | below td5, overconfident |
| Kev 0.8B / 4B (own suite) | 0.684 / 0.852 | - | - | not directly comparable |

### ANLI R1, zero-shot

| model | acc | ECE | read |
|-------|-----|-----|------|
| random (3 classes) | 0.333 | - | floor |
| OEV round-1 student (184M) | 0.3380 | 0.1155 | chance |
| OEV MNLI specialist (184M) | 0.3360 | 0.4717 | still floor. The adversarial split resists MNLI transfer |
| OEV td5 (184M), WANLI | 0.3945 | 0.1075 | +0.06 over floor |

Kev publishes no in-domain numbers (its results are on private OOD suites), so it appears only where a public stand-in row exists. OEV's own OOD splits and converters are public, and the harness is in-repo; checkpoint artifacts and their manifest are not tracked here. Run the same splits and open a PR.

## Distillation record

Student initialized from td5; pure teacher-KL loss unless noted.

| round | recipe | typed | Banking77 | emotion zero-shot | result |
|-------|--------|------:|----------:|------------------:|--------|
| R1 | 4 teachers (mt, b77, a, b), 24k unbalanced | 0.5385 | 0.8205 | **`0.6875`** | mimicry transfers breadth, loses depth |
| R2 | + gold-CE 0.3, 5 teachers, balanced | 0.1917 (choice) | 0.0104 | 0.2905 | **collapsed to uniform** (below random). Negative result |
| R2b | pure-KL, 5 teachers, balanced | **`0.6480`** | `0.7964` | `0.6505` | **shipped generalist**. Probes all PASS, gamma 1.2 |

Weight blends (same-basin interpolation): student+td5 50/50 -> typed `0.7015`, b77 `0.7894`; student+ensemble-soup 50/50 -> typed `0.7350`, b77 `0.7601`, emotion `0.5960` (not shipped). Gamma 1.2 is a historical deployable setting on distilled students (b77 ECE `0.1198` -> `0.0427` on R1; mix-valid ECE `0.0948` on R2b); its selection provenance is not tracked here.

## Decision coherence (TESTS.md scenarios, T=1.0)

| checkpoint | score | note |
|------------|------:|------|
| mt | 4/5 | destructive case: action human_review but risk Low |
| r2b student (shipped) | 3/5 | destructive_erasure -> continue. Conservatism diluted by distillation, round-4 target |

## Evaluation protocol

| rule | practice |
| ---- | -------- |
| selection | current protocol requires model selection and gamma fitting on the validation split only; historical gamma rows with missing logs are exploratory |
| test reads | each published test number was measured once; sharpened rows marked exploratory |
| splits | Banking77 official 10,003 / 1,000 / 3,080; typed-decisions published 2,000-decision test set |
| hardware | single Tesla T4 (16 GB), fp16 inference |
| baselines | Jev and laya numbers quoted from published tables, not re-run |

## Known limitations, measured

| finding | measurement | status |
| ------- | ----------- | ------ |
| benchmark-specific fine-tuning erodes general skill | zero-shot: emotion `0.4265` vs `0.9300` fine-tuned; WANLI `0.3945` vs random `0.333` | largely fixed by distillation (emotion `0.6505` zero-shot, typed `0.6480`) |
| multi-task breadth does not help zero-shot | mt scores `0.3800` on WANLI (below td5) | distillation targets include OOD calibration |
| an answer set can contain one contradiction | mt tripped 1 of 5 coherence scenarios; r2b student 2 of 5 | documented; round-4 target |
| sarcasm is read literally | "Fantastic, broke on day one" scored `0.48` positive | encoder limitation, expected |
| mild overconfidence on junk input | unstructured garbage picks an option at `~0.39` (uniform: `0.25`) | calibration work item |
| long option lists flatten score distributions | 10-level scores spread near-uniform | multi-task checkpoint addresses this |
| English only | all training and evaluation corpora are English | roadmap |

## Practical limits of one forward pass

| limit | value | notes |
| ----- | ----- | ----- |
| options per choice question | up to 255 supported by the head; measured at `77` | each option costs its own anchor plus its tokenized text |
| score levels per question | 10 | distribution over ordered levels |
| questions per request | bounded by total packed length | each question adds instructions + options as anchor rows |
| state length | `max_len - options - instructions` tokens | at `max_len 768` with 77 options, roughly `300` tokens for state |
| context reuse | none. The packed sequence is re-encoded per request | the scaling limit vs shared-KV designs; cheap at `184M` |

The confidence field on choice answers is the max probability of the distribution, a scalar summary rather than a calibrated error rate. For gating, use the full distribution and the coverage-at-error metric from `python -m oev.benchmark_ext`.

## Architecture verification

Three probes ship with the package (`python -m oev.probes --checkpoint <ckpt>`, CPU-capable):

- **isolation** - a secret in one question's instructions must not raise another question's probability of naming it above chance
- **forgery** - adversarial option text (anchor tokens, delimiter lookalikes, JSON injection) must not change anchor scoring: exactly one per given option
- **order** - argmax stability under cyclic option rotation

Measured (2026-09-24/25):

| probe | td5 | b77 | r2b student |
|-------|-----|-----|-------------|
| isolation leak vs chance | 1.06x PASS | 0.24x PASS | 0.77x PASS |
| forgery (6 adversarial cases) | PASS all | PASS all | PASS all |
| order flip rate (6 rotations) | 0.200 | 0.400 | 0.267 |

## Run it yourself

```bash
python -m oev.convert_typed

python -m oev.train --backbone microsoft/deberta-v3-base --epochs 4 --batch-size 8 --max-len 768 --data-dir data/typed --out checkpoints_td5

python -m oev.rlcd --checkpoint checkpoints_td5/oev-tiny.pt --data-dir data/typed --epochs 2 --batch-size 8 --out checkpoints_rlcd

python -m oev.benchmark_ext --checkpoint checkpoints_rlcd/oev-tiny.pt --data-dir data/typed

python -m oev.ensemble --ckpts checkpoints_td5/oev-tiny.pt,checkpoints_rlcd/oev-tiny.pt,checkpoints_rlcd_soup/oev-tiny.pt --data-dir data/typed
```

## Training data

| dataset | source | size | role |
| ------- | ------ | ---: | ---- |
| typed-decisions train split | public benchmark | 2,000 decisions | supervised fine-tune (soft targets) + RLCD |
| Banking77 train split | PolyAI (CC BY 4.0) | 10,003 queries | supervised fine-tune |
| AG News train split | public benchmark | 120,000 rows | supervised fine-tune |
| DAIR Emotion train split | public benchmark | ~316,000 rows | supervised fine-tune |
| MNLI train sample | GLUE (nyu-mll/glue) | 12,000 pairs | NLI specialist fine-tune |
| teacher-generated decisions | OEV's own teacher pipeline | internal | soft targets for multi-task pretrain |

No outputs from Jev or laya were used at any training stage; their numbers appear strictly as evaluation-time baselines.
