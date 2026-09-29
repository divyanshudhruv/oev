# Changelog

Notable changes to OEV. Versions follow [SemVer](https://semver.org/).

## Unreleased

### Added

- Round-3 distilled generalist published as `student-r3-oev-tiny.pt` (sha256
  `1871dbbb...84aa3` on the Hub): typed `0.6895` (r2b `0.6480`), Banking77
  `0.8370` (r2b `0.7964`), emotion zero-shot `0.8650` (r2b `0.6505`). Three
  teachers, 5 domains incl. MNLI, pure KL, 27,000 cases, Kaggle T4
- ANLI R1 fine-tune of the rebuilt specialist published as
  `anli-r1-oev-tiny.pt`: first adversarial NLI row, `0.5750` in-domain
  (chance `0.333`), WANLI zero-shot `0.5690`, the project's best NLI transfer.
  Known cost: OOD overconfidence (ECE `0.2308`), calibration pending
- Round-4 distilled generalist published as `student-r4-oev-tiny.pt`
  (sha256 `a43869a3...03ce2b` on the Hub): typed `0.6985` (r3 `0.6895`),
  emotion zero-shot `0.8700` (r3 `0.8650`), Banking77 `0.8188` (r3 `0.8370`,
  stated trade for the 5 head-to-head fix domains now in training)

### Fixed

- Banking77 validation split: the converter carved the un-shuffled last 1,000
  train rows as valid, and the source csv sorts by intent, so valid held only
  8 of 77 intents (938 of 1,000 valid questions had intents unseen in train).
  Now shuffled with a fixed seed before carving and gated to span the intent
  space. Test numbers, including every published Banking77 figure, were never
  affected
- `mnli-oev-tiny.pt` on the Hub replaced: it held multi-task weights from an
  upload mistake; it is now the rebuilt MNLI specialist (WANLI `0.5265`,
  sha256 `ff9095...ba5`). The lost original scored `0.5645`
- ECE `0.0279` claim attribution corrected to `oev-base-rlcd-soup`, matching
  the BENCHMARKS table and the model card

### Changed

- `oev/convert_banking77.py` shuffle + intent-coverage gate
- Charts: zeroshot and transfer panels updated for the ANLI checkpoint and the
  round-3 emotion zero-shot; registry cross-checks extended

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
