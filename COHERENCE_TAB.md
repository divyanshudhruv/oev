# Coherence demo tab

The Playground now runs a post-hoc policy layer after each decision. It compares
selected answer fields and reports consistent or contradictory findings below the
probability bars. It does not change model outputs or claims calibration.

## Rules

- `action=human_review` expects `needs_review=yes`.
- `action=stop` expects `needs_review=yes`.
- `action=continue` expects `needs_review=no`.

Yes/no comparisons use `p_yes >= 0.5` as the decision threshold. Choice
comparisons use exact option matching. Findings are escaped before rendering.

## Scope

The rules are an auditable application policy layer, not a learned coherence
metric. A missing field is skipped. A future rule change should add a focused
unit test and a recorded scenario result before changing published claims.
