# AGENTS.md

Conventions for AI agents and humans working in this repository.

## Layout

- `oev/` - the package: model, dataset, training (train, rlcd), inference
  (infer), eval (benchmark, benchmark_ext, calibrate), converters
  (convert, convert_typed, convert_banking77), serving (serve, presets),
  ensembling (ensemble, soup), probes (safety/order checks), distillation
  (distill)
- `tests/` - pytest suite (65 tests), CPU only, no checkpoints required;
  includes app-UI tests (`tests/test_app.py`)
- `examples/` - worked usage scripts
- `app.py` - the Hugging Face Space demo (Gradio); syncs to the Space on
  every push to main
- `scripts/` - chart generators (`make_charts.py`, `make_bigfigure.py`);
  all charts are transparent-background pastel with light + dark variants
- `docs/` - private working notes (gitignored); only `claims.json` is
  tracked, mapping every published number to its checkpoint and eval command
- `assets/` - README charts and logo; regenerate with
  `python scripts/make_charts.py` rather than editing PNGs by hand
- `ROADMAP.md` - forward work list, linked from the README roadmap section

## Rules

1. Never commit or push without the owner's explicit approval. Checkpoint
   files (`*.pt`), `data/`, and private session notes (`KAGGLE.md`,
   `SUMMARY.md`, `OEVIA_SPEC.md`, `docs/` except `claims.json`) are
   gitignored and must never be committed.
2. Commit messages follow `type(scope): description` with a lowercase scope
   (e.g. `fix(b77): ...`, `docs(bench): ...`, `chore(ruff): ...`). Check
   `git log` first and match. No signatures or trailers in commit messages.
3. Run `python -m pytest -q` before proposing any code change. The suite
   is CPU-only and finishes in under a minute.
4. Docstrings are banned in this codebase by owner preference. Use plain
   comments where explanation is needed.
5. No em dashes, curly quotes, or other typography that renders as
   AI-generated in committed markdown. No `statement - appositive; fragment`
   sentence patterns either; write plain sentences with varied structure.
6. Benchmark numbers come from measured runs (see `docs/claims.json` for
   checkpoint and eval provenance). Never invent, extrapolate, or round
   numbers in docs. T4 eval runs are fp32 (no autocast in the eval paths).
7. Model selection and any hyperparameter (including gamma) are chosen on
   the validation split only. Test sets are read once per published claim.
   Historical gamma `2.5` results predate this rule and are labeled
   exploratory wherever they appear.
8. When editing `oev/train.py`, remember validation runs must execute
   under no_grad with fp16 autocast or they will OOM on a 16 GB GPU.

## Environment

- Python 3.10+ (see `.python-version`)
- `pip install -e ".[dev]"` for tests, `".[backbone]"` for training,
  `".[serve]"` for the HTTP server, `".[app]"` for the Space demo
- GPU work runs on Kaggle T4 or Colab sessions; quota is limited, so prefer
  warm-started fine-tunes over from-scratch runs
