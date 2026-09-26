# Changelog

Notable changes to OEV. Versions follow [SemVer](https://semver.org/).

## 0.3.0

### Added

- Jev/TypeSafe `criteria` question schema accepted everywhere: `oev.infer.OEV.decide`,
  the `/v1/systemone` server and the Space normalize it to the native format, so
  existing laya/Kev/TypeSafe clients work by changing the base URL alone
- `oev.OEV` re-export, so `from oev import OEV` works alongside
  `from oev.infer import OEV`
- MNLI specialist trained: WANLI transfer reaches `0.5645`, ANLI stays at
  chance. The Colab artifact was lost before download, so the number is
  documented without a public checkpoint; a rebuild is on the roadmap
- Auto-releases: tag pushes publish to PyPI and create the GitHub release
- Security policy (`SECURITY.md`)

### Changed

- ONNX INT8 CPU deployment measured: `54.2 ms` p50 on 8 threads, `228 MB`
  artifact (`scripts/bench_latency.py`)
- Roadmap carries the round-4 training data checklist from the live
  head-to-head findings

## 0.2.0

### Added

- Shipped generalist checkpoint `student-r2b-oev-tiny.pt`: typed-decisions
  `0.6480`, Banking77 `0.7964`, DAIR Emotion zero-shot `0.6505` (head-to-head
  win over laya `0.595`), safety probes all PASS
- Pure teacher-KL distillation with per-domain balanced sampling; the
  collapse-to-rescue arc is documented in BENCHMARKS.md
- Banking77 soup checkpoint at `0.8584` single-file accuracy
- ONNX export and CPU latency tooling (`scripts/export_onnx.py`,
  `scripts/bench_latency.py`)
- Rebuilt Hugging Face Space UI

### Fixed

- Packer clamps anchor positions after sequence clipping (device-side assert
  on over-budget sequences)
- Eval paths state the device explicitly; fp32 labeling matches execution
