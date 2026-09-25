# Changelog

## [0.2.0] - 2026-09-25

The distillation release: a single-file generalist, a documented collapse-to-rescue
arc, and the benchmark record behind it.

> [!WARNING]
> The `gamma 1.2` deployment note and historical sharpening values depend on
> session logs that are not tracked in this repository. Treat them as
> exploratory until the selection logs and raw measurements are archived.

### Added
- **Shipped generalist checkpoint** - `student-r2b-oev-tiny.pt` on the Hub:
  typed-decisions 0.6480, Banking77 0.7964, DAIR Emotion zero-shot 0.6505
  (head-to-head win over laya 0.595), probes all PASS, historical deploy note: gamma 1.2
- Distillation rework: pure-KL loss path (`--alpha 0`), per-domain balanced
  sampling (`--per-domain`), snapshot saves (`--save-every`), 5/6-teacher support
- `oev.convert_commands` - synthetic command-intent dataset (10k cases) with
  observe/prefetch/commit signal
- Charts: zero-shot/OOD split, latency comparison, combined `transfer_speed`
  figure (light + dark variants)
- Rebuilt HF Space UI: vertical dark layout, inputs/outputs columns, r2b as the
  default checkpoint
- ONNX export + CPU latency tooling (`scripts/export_onnx.py`, `scripts/bench_latency.py`)

### Changed
- Model card: r2b student listed as recommended default; checkpoint names synced
  to Hub filenames
- BENCHMARKS.md: full distillation record - round-1 mimicry ablation, round-2
  negative result (gold-CE collapse, documented with numbers), round-2b rescue
  (reproduced twice), b77 retune/TTA rows, MNLI specialist NLI transfer
  (WANLI 0.5645, +23 over floor), OOD protocol invitation, round-4 plan

### Known issues
- Coherence regression: r2b student scores 3/5 on the five TESTS.md scenarios
  (mt baseline 4/5). Distillation diluted decision conservatism, round-4 target
- ANLI R1 remains at chance even with NLI training (adversarial split)
- Banking77 gap to Jev stands (0.8584 vs 0.870). Rotation-TTA measured
  ineffective (+0.0013)

## [0.1.0] - 2026-09-23

Initial public release: OEV typed-decision model, benchmark harness
(`benchmark_ext`), probes (isolation/forgery/order), converters, ensemble +
soup tooling, FastAPI sidecar (`oev-serve`), and the benchmark record against
laya and Jev.
