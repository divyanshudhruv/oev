# AGENTS.md

Conventions for AI agents and humans working in this repository.

## Layout

- `oev/` - the package: model, dataset, training (train, rlcd, distill),
  inference (infer), eval (benchmark, benchmark_ext, calibrate),
  converters (convert, convert_typed, convert_banking77), serving (serve,
  presets), ensembling (ensemble, soup), probes (safety and order checks)
- `tests/` - CPU-only pytest suite, no checkpoints required; includes
  app-UI tests (`tests/test_app.py`)
- `app.py` + `space_ui/` - the Hugging Face Space demo (Gradio); syncs to
  the Space on every push to main
- `examples/` - worked usage scripts
- `scripts/` - chart generators (`make_charts.py`, `make_bigfigure.py`)
  sharing one palette and one number registry (`chartstyle.py`,
  `chartdata.py`; the registry cross-checks charted OEV numbers against
  `docs/claims.json`), plus benchmark tooling
- `docs/claims.json` - the claims registry: every published number mapped
  to its checkpoint and eval command
- `assets/` - README charts and logo; regenerate with
  `python scripts/make_charts.py` rather than editing PNGs by hand.
  `banner_typed.png` is embedded by the HF model card under that exact
  filename - do not rename or remove it
- `ROADMAP.md` - forward work list, linked from the README roadmap section

## Rules

1. Never commit or push without the owner's explicit approval. Gitignored
   local artifacts (checkpoints, data, eval runs, working notes) are never
   committed.
2. Commit messages follow `type(scope): description` with a lowercase
   scope. Check `git log` first and match the existing style. No
   signatures or trailers in commit messages.
3. Run `python -m pytest -q` before proposing any code change. The suite
   is CPU-only.
4. Docstring policy: module-level docstrings are allowed (usage, schema
   notes, CLI examples); function and class docstrings are not. Inside
   functions and classes use plain comments where explanation is needed.
5. No em dashes, curly quotes, or other typography that renders as
   AI-generated in committed markdown.
6. Benchmark numbers come from measured runs recorded in
   `docs/claims.json`. Never invent, extrapolate, or round numbers in
   docs. T4 eval runs are fp32 (no autocast in the eval paths).
7. Model selection and any hyperparameter fitting use the validation
   split only. Test sets are read once per published claim.
8. Validation passes inside `oev/train.py` must run under no_grad with
   fp16 autocast, or they will OOM on a 16 GB GPU.

## Environment

- Python 3.10+ (see `.python-version`)
- `pip install -e ".[dev]"` for tests, `".[backbone]"` for training,
  `".[serve]"` for the HTTP server, `".[app]"` for the Space demo
- CI installs from the hash-locked files in `ci/` (regenerate with
  `python ci/regen_locks.py`); GPU work runs on Kaggle T4 or Colab
