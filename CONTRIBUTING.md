# Contributing to OEV

Thanks for your interest in OEV. Contributions are welcome across the model, inference engine, benchmarks, tooling, and documentation.

## Reporting bugs

Open an issue using the bug report template. Please include:

- OS, Python version, and PyTorch version
- the exact command or code snippet that failed
- the complete error traceback
- the OEV version or commit, if relevant
- the checkpoint being used, if the issue involves a model

For checkpoint-specific issues, include the checkpoint name (for example, `oev-base-td5.pt`).

## Suggesting features

Open an issue using the feature request template.

Areas that fit particularly well with OEV include:

- new decision primitives or anchor designs
- inference and latency improvements
- calibration methods
- quantization and CPU deployment
- ONNX / export support
- benchmark integrations with reproducible baselines
- multi-question inference
- training and distillation improvements
- developer tooling and documentation

For larger changes, opening an issue first is recommended so the approach can be discussed before implementation.

## Pull requests

1. Fork the repository and create a branch from `main`.

2. Install the development dependencies:

```bash
pip install -e ".[dev]"
```

3. Run the test suite:

```bash
python -m pytest -q
```

4. Keep each PR focused on one change.

5. Add or update tests when changing model, inference, or training code.

6. If changing benchmark-related code, re-run the affected benchmark.

7. Include relevant before/after measurements in the PR description for performance or model changes.

8. Keep documentation up to date when changing public APIs or behavior.

Please avoid unrelated formatting or refactoring changes in the same PR.

### Before opening a PR

- `python -m pytest -q` passes locally
- no unrelated files are included in the diff
- commit messages follow the existing `type(scope): summary` format

## Running the test suite

```bash
pip install -e ".[dev]"
python -m pytest -q
```

The test suite runs on CPU and does not require model checkpoints or a GPU.

## Re-running benchmarks

The complete benchmark and reproduction commands are documented in [BENCHMARKS.md](BENCHMARKS.md#run-it-yourself).

When submitting new benchmark results, include:

- hardware and software versions
- exact command used
- dataset and split
- checkpoint or model configuration
- relevant metrics
- whether the result is directly comparable to the existing benchmark

Measured results from different hardware and datasets are welcome, provided they can be reproduced.

## Questions and discussions

For questions about the architecture, benchmarks, or possible contributions, open a GitHub Discussion or an issue with enough context for others to reproduce or evaluate the idea.
