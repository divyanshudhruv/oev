# Coherence demo tab - built now, enable after distillation

What it does: a post-hoc policy layer that checks whether OEV's answers
compose into one story (the TESTS.md weakness, turned into a feature).
No competitor demos this.

## Integration (when the distilled model ships)

Add to app.py after the Decide wiring:

```python
# ---- coherence check ----
COHERENCE_RULES = [
    # (if field A says X, then field B should be Y) -> (label, ok)
    ("action", "human_review", "needs_review", True),
    ("action", "stop",         "needs_review", True),
    ("action", "continue",     "needs_review", False),
]

def _coherence(result: dict) -> list:
    """Return human-readable coherence findings for one decide() result."""
    findings = []
    vals = {name: (v if isinstance(v, (int, float)) else
                   v.get("choice", v.get("value"))) for name, v in result.items()}
    probs = {name: v for name, v in result.items() if isinstance(v, dict) and "probabilities" in v}
    for if_field, if_val, then_field, then_val in COHERENCE_RULES:
        a, b = vals.get(if_field), vals.get(then_field)
        if a == if_val and b is not None:
            ok = (b >= 0.5) if isinstance(then_val, bool) and then_field in probs else (b == then_val)
            findings.append((f"{if_field}={if_val} but {then_field}={'yes' if b else b}",
                             "consistent" if ok else "CONTRADICTION"))
    return findings

def _coherence_html(findings) -> str:
    if not findings:
        return ""
    rows = "".join(
        f'<div class="bar-row"><div class="bar-label">{desc}</div>'
        f'<div class="bar-val" style="color:{"var(--color-success)" if verdict == "consistent" else "var(--color-warning)"}">'
        f'{verdict}</div><div class="bar-track"></div></div>'
        for desc, verdict in findings)
    return (f'<div class="result-card"><span class="qname">coherence check</span>'
            f'<span class="qtype"> · policy layer</span><div style="margin-top:8px">{rows}</div></div>')
```

Then in `_parse`, after `triples` are built, convert to a dict, run
`_coherence`, and prepend `_coherence_html(findings)` to the rendered cards.

## Why this design

- Rules are declarations, not ML - a policy layer any team can audit
- Findings render as cards in the existing UI vocabulary
- After distillation, the demo claim is measurable: "the distilled model
  trips 0 of 4 rules on the 5 TESTS.md scenarios, the td5 specialist tripped 2"
