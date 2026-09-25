# AGENTS.md

Conventions for AI agents and humans working in this repository.

## Layout

- `oev/` - the package: model, dataset, training (train, rlcd), inference
  (infer), eval (benchmark, benchmark_ext, calibrate), converters
  (convert, convert_typed, convert_banking77), serving (serve, presets)
- `tests/` - pytest suite, CPU only, no checkpoints required
- `examples/` - worked usage scripts
- `docs/` - supporting material for the published claims (claims.json maps
  every published number to its checkpoint and eval command)
- `assets/` - README charts and logo
- `PLAN.md` - research plan with pre-registered criteria and recorded failures
- `ROADMAP.md` - forward work list, linked from the README roadmap section

## Rules

1. Never commit or push without the owner's explicit approval. Checkpoint
   files (`*.pt`) and `data/` are gitignored and must never be committed.
2. Commit messages follow `type(scope): description` (e.g.
   `fix(b77): ...`, `docs(bench): ...`). Check `git log` first and match.
3. Run `python -m pytest -q` before proposing any code change. The suite
   is CPU-only and finishes in under a minute.
4. Docstrings are banned in this codebase by owner preference. Use plain
   comments where explanation is needed.
5. No em dashes, curly quotes, or other typography that renders as
   AI-generated in committed markdown.
6. Benchmark numbers come from runs recorded under `runs/` or a Kaggle
   session log. Never invent, extrapolate, or round numbers in docs.
7. Model selection and any hyperparameter (including gamma) are chosen on
   the validation split only. Test sets are read once per published claim.
   Session notes (SUMMARY.md, summary_kaggle.md, kaggle_multitask_cells.md)
   are private and gitignored; never commit them.
8. When editing `oev/train.py`, remember validation runs must execute
   under no_grad with fp16 autocast or they will OOM on a 16 GB GPU.

## Environment

- Python 3.10+ (see `.python-version`)
- `pip install -e ".[dev]"` for tests, `".[backbone]"` for training,
  `".[serve]"` for the HTTP server
- GPU work runs on Kaggle T4 sessions; quota is limited, so prefer
  warm-started fine-tunes over from-scratch runs
