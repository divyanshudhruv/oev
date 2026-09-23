# Contributing to OEV

Thanks for looking at the code. This is a young project and any help is welcome - bug reports, benchmark results, better docs, or code.

## Reporting bugs

Open an issue with the bug report template. Please include:

- your OS, Python version, and torch version
- the exact command or snippet that failed
- the full error traceback

If the bug involves a checkpoint, say which one (e.g. `oev-base-td5.pt`).

## Suggesting features

Open an issue with the feature template. Things that fit the project's direction:

- new question primitives or anchor designs
- benchmark integrations (especially ones with published baselines to compare against)
- quantization / deployment work
- calibration improvements

## Pull requests

1. Fork, then create a branch from `main`.
2. Run the tests: `python -m pytest -q` - everything should pass before you start.
3. Keep changes focused. One PR, one thing.
4. If you change model or training code, add or update a test.
5. If you change benchmark-related code, re-run the affected benchmark and include the before/after numbers in the PR description.

## Running the test suite

```bash
pip install -e ".[dev]"
python -m pytest -q
```

The tests run on CPU and take a couple of minutes. No checkpoints or GPU required.

## Re-running benchmarks

The full commands are in [BENCHMARKS.md](BENCHMARKS.md#reproducibility). If you benchmark OEV on new hardware or new datasets, open an issue with your results - measured numbers from other people are very welcome.
