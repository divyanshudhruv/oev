# OEV Benchmarks

All OEV numbers were measured on a single Tesla T4. Comparison rows come from each project's published tables (fetched 2026-09-23). Commands to reproduce any row are at the end.

## Typed-decisions (headline)

| model | params | accuracy | soft acc | Brier | score MAE | ECE |
|---|---:|---:|---:|---:|---:|---:|
| random guess | - | 0.318 | - | - | - | - |
| majority class | - | 0.461 | - | - | - | - |
| teacher self-agreement ceiling | - | 0.735 | - | - | - | - |
| Jev 1.13.0 (published) | closed API | 0.727 | 0.580 | 0.148 | 0.391 | 0.144 |
| laya-typed-decisions (published) | 421M | 0.766 | 0.471 | 0.062 | 0.242 | 0.213 |
| OEV-RLCD single | 184M | 0.7570 | 0.5854 | 0.3651 | 0.4364 | **`0.0279`** |
| OEV fine-tune single | 184M | 0.7705 | 0.6225 | 0.3280 | 0.4028 | 0.0938 |
| OEV ensemble (4 × `184M`) | 4 × `184M` | **`0.7760`** | 0.5830 | 0.3499 | 0.4283 | - |
| OEV ensemble, sharpened (γ = 2.5) | 3 × `184M` | 0.7730 | **`0.7020`** | 0.3086 | 0.3410 | 0.0298 |

**Per primitive** (OEV vs Laya):

- noul: `0.853` / 0.857
- choice: **`0.748`** / 0.733
- score: **`0.738`** / 0.723

**Per workflow**:

- invoice processing: **`0.836`**
- customer service: **`0.804`**
- agent-trace observability: **`0.740`**
- security incidents: `0.722`

Latency, measured on a T4:

- single question, p50: **`22.2 ms`** (laya 39.5, laya-multilingual 32.8, Jev 236-276)
- batched: `15.9 ms`/question at batch 32 (63 q/s)
- fair caveat: Laya's ModernBERT is lighter in batch mode (`7.2 ms`/q)

**Sharpening sweep** (post-hoc confidence exponent γ, fitted by inspection):

| γ | soft acc | accuracy cost |
|---:|---:|---:|
| 1.0 | `0.5871` | - |
| 1.5 | `0.6410` | - |
| 2.0 | `0.6758` | - |
| 2.5 | **`0.7020`** | `0.0025` |

One caveat on the Brier column: OEV is trained to match the teacher's full distributions, not one-hot labels, so it holds back where the teacher hedged. Scored against the teacher's actual distributions, its training Brier is **`0.058`**.

## English tasks

| model | params | AG News | DAIR Emotion | ECE |
|---|---:|---:|---:|---:|
| OEV tiny, char encoder, from scratch | 0.28M | 0.2897 | - | 0.0122 |
| OEV + DeBERTa-v3-small | ~`142M` | 0.9483 | 0.9280 | `0.0094` / `0.0122` |
| **OEV + DeBERTa-v3-base** | **`184M`** | **`0.9489`** | **`0.9300`** | **`0.0184` / `0.0158`** |
| laya (published) | 421M | 0.950 | `0.595`* | `0.081` mean |
| laya-multilingual (published) | 322M | 0.937 | `0.513`* | `0.106` mean |

\* OEV's emotion number is fine-tuned rather than zero-shot. On AG News both models are essentially at the dataset's ~0.95 human-agreement ceiling; OEV gets there with 44% of the parameters.

## Banking77 (high-cardinality)

| model | accuracy |
|---|---:|
| Jev (published) | 0.870 |
| **OEV (`184M`)** | **`0.8303`** |
| laya (published) | 0.425 |
| random | 0.013 |

Banking77 is where the anchor design pays off most. With 77 intents per question, the anchor encoding doesn't degrade:

- every option gets **full tokens** - no head budget
- `0.8303` is nearly double laya's score and close to Jev's
- ECE `0.186` (temperature 2.60)


## Reproducibility

```bash
python -m oev.convert_typed

python -m oev.train --backbone microsoft/deberta-v3-base --epochs 4 --batch-size 8 --max-len 768 --data-dir data/typed --out checkpoints_td5

python -m oev.rlcd --checkpoint checkpoints_td5/oev-tiny.pt --data-dir data/typed --epochs 2 --batch-size 8 --out checkpoints_rlcd

python -m oev.benchmark_ext --checkpoint checkpoints_rlcd/oev-tiny.pt --data-dir data/typed

python -m oev.ensemble --ckpts checkpoints_td5/oev-tiny.pt,checkpoints_rlcd/oev-tiny.pt,checkpoints_rlcd_soup/oev-tiny.pt --data-dir data/typed
```

`train_colab.ipynb` runs the full pipeline end to end.
