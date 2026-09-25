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
| **OEV soup, single file** | **184M** | **0.8584** | 0.0965 | **0.7604** |
| OEV probability ensemble | 3 x 184M | 0.8529 | **0.0595** | - |

Each option is embedded as its own anchor, so all 77 labels receive the full packed token budget. The soup is a weight average of three warm-started members.

## Transfer and speed

| result | value | note |
| --- | ---: | --- |
| DAIR Emotion zero-shot, shipped student | 0.6505 | laya published zero-shot: 0.595 |
| DAIR Emotion, fine-tuned specialist | 0.9300 | not comparable to laya/Jev zero-shot rows |
| Single-question latency, T4 p50 | 22.2 ms | `184M` checkpoint |
| Single-question latency, CPU p50 | 447 ms | 8 threads, `184M` checkpoint (original release measurement) |
| Single-question latency, CPU p50, ONNX INT8 | 54.2 ms | same workstation, 8 threads, `228 MB` artifact; torch fp32 on the identical machine: 252.5 ms, ONNX fp32: 58.3 ms (`scripts/bench_latency.py`) |

The confidence field is the maximum probability of a choice distribution, not a calibrated error rate. Use the full distribution and coverage-at-error results for gating.

## Evaluation rules

- Model selection and gamma fitting use the validation split only. Historical gamma `2.5` results predate this rule and remain exploratory.
- Each published test number was measured once. New runs archive a JSON receipt in `runs/` with the command, resolved device and raw timing samples.
- Banking77: the official source has `10,003` train / `3,080` test with no official validation; this repo carves the last `1,000` train rows as validation, giving `9,003 / 1,000 / 3,080`.
- Typed-decisions uses the published `2,000`-decision test set.
- OEV measurements use a single Tesla T4 with fp32 inference; CPU timing uses eight threads.
- Jev and laya numbers are published baselines, not reruns.

## Scope

- Banking77: Jev's published figure uses 72 labels, OEV uses 77, so the comparison is indicative rather than controlled
- Historical gamma `2.5` rows stay out of headline claims until their selection logs are archived
- Out-of-domain transfer is strong on emotion and still open on adversarial NLI
- Training and evaluation data are English-only
- Each request re-encodes the packed state; long option lists consume token budget

## Checkpoint integrity

SHA-256 of every checkpoint published on the Hub, computed from the training artifacts and verified against the Hub's stored hashes. A download can be checked with `sha256sum <file>`.

| hub file | sha256 |
| --- | --- |
| student-r2b-oev-tiny.pt | `02f8f8c85c92e78d38f99bf50c38f7d35086c942473c5738507546e869999828` |
| oev-base-td5.pt | `d9e93e99af60f2263a626c4bce6136b91429d9c4f12217d9ebf3d0467888aa3a` |
| oev-base-mt.pt | `9e26803184eda11814cf9bde69b3534c3a368f5cad6619167d1c42077ce71d65` |
| oev-base-rlcd.pt | `e07fd7494c400a5306190f812c4ecdd1d006df05740f135180b5165c93d3af8f` |
| oev-base-rlcd-seed1.pt | `b4b7dc1789e3c54b7f4ba135973e2491c8215ec4721a1dc8f1fa58a898c880f2` |
| oev-base-rlcd-soup.pt | `e0e8f73de556d3b707b45a49980186ce7e5fac5c0b2e8d81f4f9b7fd901eb675` |
| b77-oev-tiny.pt | `39ad8b2fb8cbc9009b562002ed81336f41e4f9e2c0b6a350011471ffe5a7e1e2` |
| b77a-oev-tiny.pt | `7c478201bab16314bb84a01dd0de21fda161830c0e09a957f4ec9bccb20bf720` |
| b77b-oev-tiny.pt | `f64384fae895d534abfbac391069ce2afd3c96dab79f742031526fdd830777b6` |
| b77soup-oev-tiny.pt | `eb9d495a84669edd6a03ad23e4037516839f4fcf6f475dd7d2522a4a251b1274` |

## Reproduce

```bash
python -m oev.convert_typed
python -m oev.benchmark --checkpoint checkpoints_td5/oev-tiny.pt --data-dir data/typed
python -m oev.benchmark_ext --checkpoint <checkpoint> --data-dir data/banking77 --latency
```

Training uses public benchmark splits plus OEV teacher-generated soft targets. No Jev or laya outputs were used for training.
