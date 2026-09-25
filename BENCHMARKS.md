# OEV Benchmarks

OEV results use a single Tesla T4 (16 GB) with fp16 inference unless stated otherwise. Jev and laya values are quoted from their published tables and were not rerun.

> [!WARNING]
> Banking77 uses 77 OEV labels, while the published Jev result uses 72 labels. The Banking77 values are not a controlled head-to-head comparison. Historical gamma `2.5` results are exploratory and are not release claims because the validation-selection log is unavailable. Checkpoint hashes and raw run manifests are not tracked in this repository.

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
| Single-question latency, T4 p50 | 22.2 ms | `184M` checkpoint |
| Single-question latency, CPU p50 | 447 ms | 8 threads, `184M` checkpoint |

The confidence field is the maximum probability of a choice distribution, not a calibrated error rate. Use the full distribution and coverage-at-error results for gating.

## Evaluation rules

- Model selection and gamma fitting use the validation split only.
- Each published test number was measured once.
- Banking77 uses the official `10,003 / 1,000 / 3,080` train, validation, and test splits.
- Typed-decisions uses the published `2,000`-decision test set.
- OEV measurements use a single Tesla T4 with fp16 inference; CPU timing uses eight threads.
- Jev and laya numbers are published baselines, not reruns.

## Limitations

- Banking77 comparisons are not controlled because the published Jev result uses 72 labels and OEV uses 77.
- Historical gamma `2.5` results are omitted from headline claims because validation provenance is incomplete.
- Out-of-domain transfer is mixed: emotion transfer is positive, while adversarial NLI transfer remains near chance.
- Training and evaluation data are English-only.
- The model re-encodes the packed state for each request; long option lists consume token budget.

## Reproduce

```bash
python -m oev.convert_typed
python -m oev.benchmark --checkpoint checkpoints_td5/oev-tiny.pt --data-dir data/typed
python -m oev.benchmark_ext --checkpoint <checkpoint> --data-dir data/banking77 --latency
```

Training uses public benchmark splits plus OEV teacher-generated soft targets. No Jev or laya outputs were used for training.
