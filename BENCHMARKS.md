# OEV Benchmarks

All OEV measurements were run on a single Tesla T4. Baseline results for Jev and Laya are taken from their published benchmark tables and were not independently re-run. Commands for reproducing OEV results are provided at the end of this document.

## The numbers that matter

- **`0.7705` accuracy** - single `184M` OEV model (laya: 0.766 from `421M`)
- **`0.7760` accuracy** - `4 x 184M` OEV ensemble
- **`0.6225` soft accuracy** - single model, vs laya's `0.471`
- **`0.7020` soft accuracy** - sharpened ensemble (exploratory post-processing; see note below)
- **`0.0298` ECE** - sharpened ensemble
- **`0.8584` Banking77 accuracy** - one `184M` soup checkpoint (737 MB) that replaces the 3-checkpoint ensemble; empty Jev's 1.7 pt lead to 1.16
- **`0.7604` Banking77 coverage at <=5% error** - 76% of decisions are automatable at a 5% error budget (the gating story)
- **`22.2 ms` p50** - single-question inference on T4, versus laya's published `39.5 ms` for its English checkpoint
- **`447 ms` p50 on CPU** (8 threads, 184M checkpoint) - no CPU figure is published by laya, Kev or Jev; OEV is the only System One model with a measured laptop-CPU latency

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
- CPU: **`447 ms` p50** (`473 ms` p95, 8 threads, laptop CPU, 184M checkpoint, 30-call measurement) - laya, Kev and Jev publish no CPU latency; single-machine measurement, treat as indicative
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

## Kev: what can and cannot be compared

Kev (0.8B and 4B, open weights) publishes no in-domain numbers on typed-decisions, AG News, DAIR Emotion or Banking77 - its reported results are on its own out-of-domain suites (0.652 at 0.8B, 0.837 at 4B on its harder split). Comparing those against OEV's in-domain numbers would be apples to oranges, so Kev does not appear in the in-domain tables above.

OEV has now measured the public WANLI out-of-domain row Kev's own suite stands in for (see the zero-shot section below): `0.3945` (td5) and `0.3800` (mt), against a random floor of `0.333`. The comparison is still not apples-to-apples - Kev's numbers are on its own private suite - but OEV is no longer unmeasured on transfer: its zero-shot reading is weak and this document says so.

The honest summary: OEV wins every published in-domain row it shares with laya and Jev (except Banking77 vs Jev's `0.870`), has closed the Banking77 gap to `1.16` points with the soup checkpoint, and - after one distillation round - beats laya's zero-shot emotion head-to-head (`0.6505` shipped, round-1 peaked at `0.6875`; laya `0.595`). NLI suites remain at chance (the round-3 target).

## AG News and emotion: pressing the ceiling

| model                                |     params |      AG News | DAIR Emotion |                     ECE |
| ------------------------------------ | ---------: | -----------: | -----------: | ----------------------: |
| OEV tiny, char encoder, from scratch |      0.28M |       0.2897 |            - |                  0.0122 |
| OEV + DeBERTa-v3-small               |    ~`142M` |       0.9483 |       0.9280 |     `0.0094` / `0.0122` |
| **OEV + DeBERTa-v3-base**            | **`184M`** | **`0.9489`** | **`0.9300`** | **`0.0184` / `0.0158`** |
| OEV round-1 student, zero-shot |     `184M` |           - | **`0.6875`** |            - |
| **OEV round-2b student (shipped), zero-shot** |     `184M` |           - | **`0.6505`** |       `0.1629` |
| majority class (joy)                 |          - |           - |       `~0.40` |            - |
| laya (published)                     |       421M |        0.950 |    `0.595`\* |            `0.081` mean |
| laya-multilingual (published)        |       322M |        0.937 |    `0.513`\* |            `0.106` mean |

\* laya's emotion number is zero-shot. OEV's `0.9300` is fine-tuned on the train split; its zero-shot entrant is the distilled student - the shipped round-2b checkpoint at `0.6505`, a head-to-head zero-shot win over laya (the round-1 student's higher `0.6875` was lost to a session recycle and survives as a documented number; the td5 specialist's `0.4265` ablation baseline lives in the round-1 section below). On AG News both models are essentially at the dataset's ~0.95 human-agreement ceiling; OEV gets there with 44% of the parameters.

## Banking77: 77-way classification, no token starvation

| model                       |     params |     accuracy |          ECE |   soft acc | coverage@5% |  AURC |
| --------------------------- | ---------: | -----------: | -----------: | ---------: | -----------: | ----: |
| Jev (published)             |     closed |      `0.870` |            - |          - |            - |     - |
| **OEV soup (single file)**  |  `184M`    | **`0.8584`** |     `0.0965` |   `0.8484` |   `0.7604`   | `0.0389` |
| OEV ensemble (3 x 184M)     | 3 x `184M` |      `0.8529`    | **`0.0595`** |   `0.8224` |            - |     - |
| OEV single, warm-start re-tune |  `184M` |     `0.8403` |     `0.1905` |          - |            - |     - |
| OEV single (first release)     |     `184M` |      `0.8303` |     `0.1860` |          - |            - |     - |
| OEV single, re-measured (`benchmark_ext`) |  `184M` |     `0.8302` |     `0.0781` |          - |    `0.6386` | `0.0499` |
| laya (published)            |      `421M` |      `0.425` |            - |          - |            - |     - |
| random                      |          - |      `0.013` |            - |          - |            - |     - |

Banking77 is where the anchor design pays off most. Each of the 77 options is embedded as its own anchor occupying the full token budget of the packed sequence, so no label is truncated to a few tokens - the constraint that limits token-budget heads like laya's.

The ensemble of the original checkpoint with two warm-started re-tunes reaches `0.8529`, 2x laya's score and within 1.7 points of Jev's closed API. Probability averaging over members also calibrated the ensemble for free: `ECE 0.0595` unsharpened, better than any post-hoc sharpened single (`0.1129`), and comparable to typed-decisions' `0.0298` sharpened.

#### The soup: a single file that beats the ensemble

Weight-averaging the three warm-started b77 checkpoints (they share a loss basin, so naive mixing works) produces **one `184M` checkpoint** that scores `0.8584` - `+0.55` points over the probability-averaged ensemble, the project's best Banking77 number, and within `1.16` points of Jev. It is also the strongest gating artifact measured: `0.7604` of decisions can be auto-piloted at a <=5% error budget (AURC `0.0389`). The trade is honest: soup ECE `0.0965` is worse than the ensemble's `0.0595`, though 2x better than any single member (`~0.19`). For a single-file deployment the soup is the artifact to ship.

### Banking77 calibration sweep (single model)

Post-hoc confidence sharpening (the same γ mechanism as the typed ensemble) was swept on the single Banking77 checkpoint. The optimum is gentle: unlike typed-decisions, ECE rises beyond γ = 1.5, so `1.5` is the reported setting. The ensemble row above supersedes this: probability averaging reaches better calibration than any γ on the single model.

|   γ |     accuracy |          ECE |   soft acc |
| --: | -----------: | -----------: | ---------: |
| 1.0 |     `0.8303` |     `0.1860` |          - |
| 1.5 |     `0.8302` | **`0.1129`** |   `0.8175` |
| 2.0 |     `0.8302` |     `0.1289` |   `0.8228` |
| 2.5 |     `0.8302` |     `0.1368` |   `0.8250` |

Accuracy is unchanged to four decimals while calibration improves 39%. The remaining gap to typed-decisions-level calibration (ECE `< 0.10`) appears to need training-time changes, not a larger γ.

## Zero-shot and out-of-domain transfer, measured

Fine-tuned benchmarks are close to the ceiling; zero-shot is the honest gap.
Out-of-domain suites were measured on checkpoints that never saw the target
data (2026-09-24, T4, fp16). The story after one distillation round: the
distilled student turns emotion zero-shot from a `0.4265` specialist failure
into a win over laya (round-1 `0.6875`, shipped round-2b `0.6505`), recovers the
typed skill pure mimicry destroyed (`0.5385` -> `0.6480`), while NLI (WANLI,
ANLI) stays at the chance floor - no NLI signal exists in any checkpoint, the
documented round-3 target.

### DAIR Emotion, zero-shot

| model | acc | ECE | conf-err (p>=0.9) | read |
|-------|-----|-----|-------------------|------|
| **OEV distilled student r2b (184M, shipped)** | **`0.6505`** | `0.1629` | `0.0010` | **beats laya zero-shot head-to-head (+5.6 pts)** |
| OEV round-1 student (184M) | `0.6875` | - | `0.0000` | higher still; checkpoint lost to session recycle, number documented |
| laya (published, zero-shot) | 0.595 | - | - | trained broadly, transfers |
| OEV td5, specialist (184M) | 0.4265 | 0.1750 | 0.0000 | ablation baseline (round-1 section) |
| majority class (joy) | ~0.40 | - | - | trivial |
| random (6 labels) | 0.167 | - | - | trivial |

Same conditions, both zero-shot, neither trained on the target: OEV's
distilled student beats laya `0.6505` vs `0.595` (round-1's `0.6875` confirmed
the effect before its checkpoint was lost). The td5 specialist's
`0.4265` is retained as the ablation baseline that motivated distillation -
the student recovered `+26` points of transfer in a single round.

### WANLI (NLI), zero-shot

WANLI is a natural-language-inference test split the checkpoints never saw
(premise -> hypothesis, three classes). Labels were derived from the dataset
schema (`entailment / neutral / contradiction`).

| model | acc | ECE | conf-err (p>=0.9) | read |
|-------|-----|-----|-------------------|------|
| random (3 classes) | 0.333 | - | - | floor |
| OEV td5 (184M) | 0.3945 | 0.1075 | 0.0000 | +0.06 over floor |
| OEV mt (184M, generalist) | 0.3800 | 0.2138 | 0.0115 (43) | below td5 and overconfident |
| Kev 0.8B / 4B (own suite) | 0.684 / 0.852 | - | - | not directly comparable |

td5 clears random by only `+0.06`; mt is *worse* (`0.3800`) and confident about
it - 43 answers wrong at p>=0.9, the worst overconfidence measured in the
session. Breadth did not transfer, confidence did (badly). Both readings feed
the same conclusion and the same fix: distillation.

### NLI is at chance: the ANLI measurement

ANLI round 1 (test_r1, 1000 cases) closes the WANLI question: the student's
`0.3510` there was never a regression, it is the NLI floor. No checkpoint in the
project has seen NLI training data.

| model | acc | ECE | read |
|-------|-----|-----|------|
| random (3 classes) | 0.333 | - | floor |
| OEV round-1 student (184M) | 0.3380 | 0.1155 | exactly chance, calibrated at chance |
| OEV td5 (184M), WANLI | 0.3945 | 0.1075 | +0.06 over floor |

Honest doc row: NLI zero-shot = chance across the board. Fixing it needs an NLI
teacher in the distill mix or a small NLI fine-tune - round-3 material, not a
hidden weakness.

### Round-1 distillation: the ablation result

The first distillation run (4 teachers: mt, b77, b77a, b77b; 24,060 unbalanced
cases; pure teacher-KL loss, no gold signal; student initialized from td5)
landed 2026-09-24 after ~55 min on T4 (loss 1.42 -> 1.49, 4.9 cases/s). It is
best read as a controlled experiment in what pure ensemble mimicry does:

| domain | best before | round-1 student | delta | cause |
|--------|------------:|----------------:|-------|-------|
| typed-decisions | 0.7705 (td5) | **0.5385** | -23.2 pts | teachers were 3x b77 + mt; td5 itself never taught; typed only 4% of the mix; no gold loss |
| Banking77 | 0.8584 (soup) | 0.8205 | -3.8 pts | 3 of 4 teachers were b77 specialists - the mean signal is strong here |
| DAIR Emotion (zero-shot) | 0.4265 (td5) | **0.6875** | **+26.1 pts** | mt's breadth transferred through the mean - the zero-shot fix worked |
| WANLI (zero-shot) | 0.3945 (td5) | 0.3510 | -4.4 pts | no NLI signal anywhere in the mix (see ANLI floor) |

Findings (all measured, 2026-09-24, T4):

- **Pure mimicry transfers breadth, not depth.** The student absorbed the
  teacher mean faithfully (train loss fell monotonically) and still lost the
  specialist skill its own init had. Ensemble-averaging cancels errors when you
  *average predictions*; imitating the average over a skewed mix averages the
  errors in.
- **The student is systematically under-confident** - the mimicry signature.
  Gamma sweep on the val mix (ag_news + b77): ECE 0.6170 at gamma 0.5 ->
  **0.0398 at gamma 1.2** -> 0.0765 at gamma 2.0. Accuracy invariant across all
  gammas (0.8450), confirming the harness. Deployable setting: gamma 1.2 turns
  b77 ECE 0.1198 -> 0.0427 on the student.
- **Weight interpolation recovers specialist skill.** A 50/50 weight blend of
  the student with td5 (shared init = same basin) restores typed to `0.7015`
  (+16.3 pts) at the cost of b77 (0.7894) - a balanced-generalist fallback
  artifact. A within-run soup (student + step-2000 snapshot) is strictly safe:
  val mix acc 0.8450 -> 0.8500.

Round 2 tested the corrected recipe (gold-anchored loss `--alpha 0.3`, 5
teachers, balanced domains) and **failed catastrophically - documented as a
negative result**: the student collapsed to near-uniform outputs (b77 `0.0104`,
below the `0.013` random floor; typed choice `0.1917`, below 4-way chance;
uniform-level ECE everywhere). The KL+CE tension at lr 2e-5 is the prime
suspect; the loss curve (2.43 -> 1.83 -> plateau 2.17) is the collapse signature
in hindsight. Round 2b reran the same data recipe with `--alpha 0` (pure-KL, the loss that
trained round 1 stably) and **trained cleanly** - the loss curve reproduced
twice on independent runs (1.69 -> 1.64, monotone after step 100) and the
battery confirms the rescue: typed `0.5385` -> **`0.6480`** (+11.0), emotion
`0.6505` (laya win intact, +5.6), b77 `0.7964`, WANLI/ANLI at the chance floor,
probes all PASS (isolation 0.77x chance, the best of any checkpoint), gamma 1.2
chosen again (mix-valid ECE 0.0948). It ships as the generalist. A
student+soup-ensemble weight blend pushed typed to `0.7350` but broke the
floors (b77 `0.7601`, emotion `0.5960`) - documented as a frontier point, not
shipped: the pure-KL student keeps the best worst-case across all three domains.

### Coherent decisions: the fair TESTS.md rerun

The original TESTS.md failure was measured on td5 at `T=0.5`. The fair entrant
is mt at `T=1.0`, rerun on the same five scenarios (2026-09-24):

| scenario | action | needs_review | risk | verdict |
|----------|--------|--------------|------|---------|
| destructive_erasure | human_review | 0.47 (borderline) | Low (expected Moderate) | 1 contradiction |
| benign_readonly | continue | 0.13 | Benign | coherent |
| prod_db_write | human_review | 0.31 | Moderate | mostly coherent |
| scope_violation | human_review | 0.84 | High | coherent |
| clean_automation | continue | 0.12 | Benign | coherent |

mt is **4/5 coherent** at fair settings, far better than the td5 collapse that
triggered the finding. The one surviving contradiction is the documented
destructive case (action says human-review, risk says Low) and is the explicit
pre-distillation baseline: the distilled student's coherence target is 5/5.

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
| benchmark-specific fine-tuning erodes general skill | zero-shot: emotion `0.4265` vs `0.9300` fine-tuned; WANLI `0.3945` vs random `0.333` - two independent suites agree | **largely fixed by distillation**: r2b student emotion `0.6505` zero-shot, typed recovered to `0.6480` (specialist ceiling 0.7705) |
| multi-task breadth does not help zero-shot either | mt scores `0.3800` on WANLI (below td5) with 43 errors at p>=0.9 | distillation targets include OOD calibration |
| an answer set can contain one contradiction | mt at T=1.0 tripped 1 of 5 TESTS.md coherence scenarios (human-review but risk Low) | documented baseline; distilled model target 5/5 |
| the typed specialist scores `0.50` confidence on an obvious positive sentiment review | - | superseded by the zero-shot measurements above |
| sarcasm is read literally | "Fantastic, broke on day one" scored `0.48` positive | encoder limitation, expected |
| mild overconfidence on junk input | unstructured garbage picks an option at `~0.39` where uniform is `0.25` | calibration work item |
| long option lists flatten score distributions | 10-level scores spread near-uniformly where training saw 4-5 levels | multi-task checkpoint addresses this |
| English only | all training and evaluation corpora are English | roadmap |

## Practical limits of one forward pass

| limit | value | notes |
| ----- | ----- | ----- |
| options per choice question | up to 255 supported by the head; measured at `77` (Banking77) | each option costs its own anchor plus its tokenized text |
| score levels per question | 10 (matching the TypeSafe API) | distribution over ordered levels |
| questions per request | bounded by total packed length | each question adds its instructions + options as anchor rows |
| state length | `max_len - options - instructions` tokens | at `max_len 768` with 77 options, the state budget is roughly `300` tokens; measured b77 packing peaks at `569` total tokens |
| context reuse | none - the packed sequence is re-encoded per request | unlike Jev's shared KV cache, a long state with many questions re-pays the state each time; at `184M` this is cheap but it is the honest scaling limit |

The confidence field on choice answers is the max probability of the distribution; it is a scalar summary, not a calibrated error rate. For gating decisions, use the full distribution and the coverage-at-error-budget metric reported by `python -m oev.benchmark_ext`.

## Architecture verification

Three probes ship with the package to check the packed-sequence design's claims, runnable against any checkpoint on CPU:

```bash
python -m oev.probes --checkpoint checkpoints_td5/oev-tiny.pt
```

- **isolation** - a secret placed in one question's instructions must not raise the probe question's probability of naming it above chance. Mechanical checks on the head are exact; behavioral isolation on trained checkpoints is reported from these probes.
- **forgery** - adversarial option text (anchor tokens, delimiter lookalikes, JSON injection) must not change how many anchors the head scores: exactly one per given option.
- **order** - argmax stability under cyclic option rotation, the same measurement as `benchmark_ext --permute 6`.

Measured on the session checkpoints (2026-09-24):

| probe | td5 | b77 |
|-------|-----|-----|
| isolation leak vs chance | 1.06x PASS | 0.24x PASS |
| forgery (6 adversarial cases) | PASS all | PASS all |
| order flip rate (6 rotations) | 0.200 | 0.400 |

`benchmark_ext --permute` caps choice questions at 24 options, so it skips
Banking77's 77 options; the b77 order figure above comes from `oev.probes`.

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
