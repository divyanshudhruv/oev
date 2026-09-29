<!-- Thank you for contributing to OEV. Keep each PR focused on one change. -->

## Summary

<!-- What does this PR change and why? -->

## Changes

- [ ] Code change (model, inference, training, tooling)
- [ ] Documentation update
- [ ] Benchmark or measurement update
- [ ] Tests added or updated

## Checks

- [ ] `python -m pytest -q` passes locally (CPU-only suite, no checkpoints needed)
- [ ] `ruff check oev scripts tests` is clean
- [ ] `python scripts/check_claims.py` passes if any published number or version changed
- [ ] No checkpoint files (`*.pt`), `data/`, or private session notes are included in the diff
- [ ] Commit messages follow `type(scope): summary`
- [ ] No em dashes, curly quotes, or fabricated benchmark numbers

## Benchmark note

<!-- If this PR changes model or performance code, include before/after numbers with hardware and command. Otherwise write "not applicable". -->
