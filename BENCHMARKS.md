# OEV Benchmarks

OEV results use a single Tesla T4 (16 GB) with fp32 inference unless stated otherwise. Jev and laya values are quoted from their published tables and were not rerun.

> [!WARNING]
> Banking77 uses 77 OEV labels, while the published Jev result uses 72 labels. The Banking77 values are not a controlled head-to-head comparison. Historical gamma `2.5` results are exploratory and are not release claims because the validation-selection log is unavailable. Benchmark runs write per-run receipts (command, device, metrics, raw timing samples) to `runs/`; runs predating this protocol have no archived samples.

## Typed decisions

| model | parameters | accuracy | soft accuracy | ECE |
| --- | ---: | ---: | ---: | ---: |
| Jev 1.13.0 (published) | closed API | 0.727 | 0.580 | 0.144 |
| laya-typed-decisions (published) | 421M | 0.766 | 0.471 | 0.213 |
| **OEV single** | **184M** | **0.7705** | **0.6225** | 0.0938 |
| **OEV ensemble** | **4 x 184M** | **0.7760** | 0.5830 | - |

The OEV ensemble combines independent `184M` checkpoints. The single model is the primary single-model result.

## Banking77

| model | parameters | accuracy | ECE | coverage at <=5% error |
| --- | ---: | ---: | ---: | ---: |
| Jev (published) | closed | 0.870 | - | - |
| laya (published) | 421M | 0.425 | - | - |
| **OEV soup4, single file** | **184M** | **0.8594** | 0.1062 | **0.7604** |
| OEV weight soup, single file | 184M | 0.8584 | 0.0965 | - |
| OEV 4-member ensemble | 4 x 184M | 0.8568 | **0.0583** | - |
| OEV probability ensemble | 3 x 184M | 0.8529 | 0.0595 | - |

Each option is embedded as its own anchor, so all 77 labels receive the full packed token budget. The soup is a weight average of warm-started members; `soup4` adds a fourth member re-tuned from the soup itself (2026-09-29, receipt `runs/20260929-125136-b77soup4-oev-tiny.json`). Ensemble saturation: 3-member 0.8529, 4-member 0.8568, 5-member 0.8542 - adding a drifted member hurts, so the recipe tops out at soup4's 0.8594. Jev's 1.06-point lead stands (72-label caveat).

## Transfer and speed

| result | value | note |
| --- | ---: | --- |
| DAIR Emotion zero-shot, round-3 student | 0.8650 | laya published zero-shot: 0.595; shipped r2b student was 0.6505 |
| DAIR Emotion, fine-tuned specialist | 0.9300 | not comparable to laya/Jev zero-shot rows |
| Single-question latency, T4 p50 | 22.2 ms | `184M` checkpoint |
| Single-question latency, CPU p50 | 447 ms | 8 threads, `184M` checkpoint (original release measurement) |
| Single-question latency, CPU p50, ONNX INT8 | 54.2 ms | same workstation, 8 threads, `228 MB` artifact; torch fp32 on the identical machine: 252.5 ms, ONNX fp32: 58.3 ms (`scripts/bench_latency.py`) |
| fp16 autocast eval, T4 p50 | 28.89 ms | accuracy identical to fp32 (0.7705 both) on the full test set; same-session fp32 p50 32.61 ms (`runs/20260928-091824-fp16eval-oev-base-td5.json`) |
| 4 questions batched, T4 | 72.17 ms | per-question loop on the same questions: 98.54 ms; bounds the shared-state encoder win (`runs/20260928-091830-batched-latency-oev-base-td5.json`) |

The confidence field is the maximum probability of a choice distribution, not a calibrated error rate. Use the full distribution and coverage-at-error results for gating.

## Calibration and out-of-distribution behavior

Temperature scaling fitted on the typed-decisions validation split only (eval-rules compliant) cuts test ECE from `0.0938` to `0.0204` on the 2,000-decision test set, beating the published best calibrated single (`0.0279`, rlcd-soup). Accuracy is unchanged by monotone temperature scaling. Use `OEV(checkpoint, temperature=0.598)` (`runs/20260928-092030-calibration-oev-base-td5.json`).

The shipped generalist (student-r2b) shows the expected mild OOD conservatism rather than overconfidence: `0` of `800` WANLI zero-shot answers and `0` of `6` garbage-input answers reach `p >= 0.9`, and on emotion `14` of `800` are confident with `1` confidently wrong (`runs/20260928-092113-ood-sweep-student-r2b.json`).

## Distilled generalist progress

The distilled generalist is the one-file model meant to cover every domain. Round 3 (pure-KL distillation, 3 teachers, 5 domains including MNLI, 27,000 cases, Kaggle T4) advanced every core number over the shipped round-2b student while keeping the single `184M` checkpoint:

| benchmark | r2b student | round-3 student | **round-4 student** | specialist | r4 receipt |
| --- | ---: | ---: | ---: | ---: | --- |
| typed-decisions | 0.6480 | 0.6895 | **0.6985** | 0.7705 | `runs/20260929-120027-student.json` |
| Banking77 | 0.7964 | 0.8370 | **0.8188** | 0.8584 | `runs/20260929-120430-student.json` |
| DAIR Emotion zero-shot | 0.6505 | 0.8650 | **0.8700** | 0.9300 (fine-tuned) | `runs/20260929-120529-student.json` |

The round-3 checkpoint is published as `student-r3-oev-tiny.pt` (sha256 `1871dbbb70567c6ce984e54a1d8d06ec15b7c3dbb0487a3ba33964aa08b84aa3`).

Round 4 warm-starts from r3 and mixes in 10 domains: the 5 core benchmarks plus 5 head-to-head fix sets (billing-vs-tech, multi-issue, benign logins, sarcasm, latency traces). It lands the best generalist typed and emotion rows, and trades `1.8` Banking77 points for failure modes the three benchmarks cannot see (the fix domains are not b77 classes). Published as `student-r4-oev-tiny.pt` (sha256 `a43869a3c9af6e8915ab4a1786202c26e67fc710aece215a82a1b2eefa03ce2b`). Both round-3 and round-4 students inherit the distillation underconfidence signature: no answer crosses `p >= 0.9`, so ECE reads high until gamma sharpening is applied (the round-1 note in docs/claims.json documents the sweep recipe).

## NLI transfer

The rebuilt MNLI specialist (12k MNLI train cases, 1 epoch, Kaggle T4) scores `0.5265` WANLI zero-shot (`runs/20260929-083252-wanli-mnli-oev-tiny.json`); the lost original scored `0.5645` on weights that no longer exist, and the rebuilt Hub file replaced the multi-task weights it previously held.

Fine-tuning that specialist on ANLI R1's train split (2 epochs, 16,946 cases) lands the project's first adversarial NLI row: `0.5750` in-domain against a `0.333` chance floor (`runs/20260929-093108-oev-tiny.json`). The same checkpoint lifts WANLI transfer to `0.5690` (`runs/20260929-093209-oev-tiny.json`), above every prior WANLI number in the project, including its own specialist init (`0.5265`) and the lost original (`0.5645`). The cost is OOD overconfidence: on WANLI the ANLI checkpoint posts ECE `0.2308` with `12.9%` of all answers landing confidently wrong (258 of 2000), so gamma/calibration on the NLI line stays open.

## Evaluation rules

- Model selection and gamma fitting use the validation split only. Historical gamma `2.5` results predate this rule and remain exploratory.
- Each published test number was measured once. New runs archive a JSON receipt in `runs/` with the command, resolved device and raw timing samples.
- Banking77: the official source has `10,003` train / `3,080` test with no official validation; this repo carves `1,000` train rows as validation after a fixed-seed shuffle, giving `9,003 / 1,000 / 3,080` with all 77 intents present in every split. (The split was previously the un-shuffled last `1,000` train rows, which sorted by intent and left valid with only 8 of 77 intents - fixed 2026-09-29. Historical checkpoint selection used that broken valid split; test numbers, including every published Banking77 figure, were never affected.)
- Typed-decisions uses the published `2,000`-decision test set.
- OEV measurements use a single Tesla T4 with fp32 inference; CPU timing uses eight threads.
- Jev and laya numbers are published baselines, not reruns.

## Scope

- Banking77: Jev's published figure uses 72 labels, OEV uses 77, so the comparison is indicative rather than controlled
- Historical gamma `2.5` rows stay out of headline claims until their selection logs are archived
- Out-of-domain transfer is strong on emotion, WANLI reaches `0.5690` through the ANLI fine-tune, and OOD overconfidence on the NLI checkpoints remains open (calibration pending)
- Training and evaluation data are English-only
- Each request re-encodes the packed state; long option lists consume token budget

## Checkpoint integrity

SHA-256 of every checkpoint published on the Hub, computed from the training artifacts and verified against the Hub's stored hashes. A download can be checked with `sha256sum <file>`.

| hub file | sha256 |
| --- | --- |
| student-r2b-oev-tiny.pt | `02f8f8c85c92e78d38f99bf50c38f7d35086c942473c5738507546e869999828` |
| mnli-oev-tiny.pt | `ff90953254e867f7b0a8eb9b3a531fd3215046ac165e2d64ad339cf8eeba8fa5` - rebuilt specialist (Kaggle T4, 2026-09-29), WANLI zero-shot `0.5265` with receipt in `runs/`. The lost original scored `0.5645`; the old Hub file held multi-task weights and was replaced by this upload |
| anli-r1-oev-tiny.pt | `ef5386e8ef659a2659e475f67428ff753c0e71ba89992e4ab5ab9aa8d1ed918e` - ANLI R1 fine-tune of the rebuilt specialist (Kaggle T4, 2026-09-29): `0.5750` in-domain, WANLI zero-shot `0.5690`, receipts in `runs/` |
| oev-base-td5.pt | `d9e93e99af60f2263a626c4bce6136b91429d9c4f12217d9ebf3d0467888aa3a` |
| oev-base-mt.pt | `9e26803184eda11814cf9bde69b3534c3a368f5cad6619167d1c42077ce71d65` |
| oev-base-rlcd.pt | `e07fd7494c400a5306190f812c4ecdd1d006df05740f135180b5165c93d3af8f` |
| oev-base-rlcd-seed1.pt | `b4b7dc1789e3c54b7f4ba135973e2491c8215ec4721a1dc8f1fa58a898c880f2` |
| oev-base-rlcd-soup.pt | `e0e8f73de556d3b707b45a49980186ce7e5fac5c0b2e8d81f4f9b7fd901eb675` |
| b77-oev-tiny.pt | `39ad8b2fb8cbc9009b562002ed81336f41e4f9e2c0b6a350011471ffe5a7e1e2` |
| b77a-oev-tiny.pt | `7c478201bab16314bb84a01dd0de21fda161830c0e09a957f4ec9bccb20bf720` |
| b77b-oev-tiny.pt | `f64384fae895d534abfbac391069ce2afd3c96dab79f742031526fdd830777b6` |
| b77soup-oev-tiny.pt | `eb9d495a84669edd6a03ad23e4037516839f4fcf6f475dd7d2522a4a251b1274` |
| b77soup4-oev-tiny.pt | `8ec4da8c93db4425749bfa73ca9fb5a08e72cf6277969021bc43e2d1101fa718` - 4-member soup, project-best b77 `0.8594` (2026-09-29, receipt in `runs/`) |

## Reproduce

```bash
python -m oev.convert_typed
python -m oev.benchmark --checkpoint checkpoints_td5/oev-tiny.pt --data-dir data/typed
python -m oev.benchmark_ext --checkpoint <checkpoint> --data-dir data/banking77 --latency
```

Training uses public benchmark splits plus OEV teacher-generated soft targets. No Jev or laya outputs were used for training.
