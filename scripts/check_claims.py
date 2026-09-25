"""Claims integrity checks. Run: python scripts/check_claims.py

Fails (exit 1) if:
- pyproject version != oev.__version__
- an OEV accuracy-like number appears in README/BENCHMARKS but not in docs/claims.json
- a claim eval command still contains a placeholder

Published competitor numbers (laya, Jev, Kev) are quoted from their tables,
not OEV claims, so they are allowlisted here.
"""
import json
import re
import sys

import tomllib

# quoted from published tables (laya / Jev / Kev), not OEV claims
COMPETITOR = {
    "0.727", "0.766", "0.595", "0.425", "0.480",   # laya + Jev benchmark rows
    "0.950", "0.910", "0.471", "0.580", "0.213", "0.144",   # laya/Jev secondary metrics
    "0.870",   # Jev banking77
    "0.648", "0.697", "0.817", "0.838", "0.822", "0.852", "0.848", "0.896",   # Kev new-sources table (dev/test)
    "0.650", "0.6500",   # laya zero-shot variant as rounded in prose
}
# protocol constants: random floors, baselines, label-noise ceilings
EXEMPT = {"0.1667", "0.333", "0.3333", "0.40", "0.4000", "0.95"}

fail = []

with open("pyproject.toml", "rb") as fh:
    py = tomllib.load(fh)["project"]["version"]
with open("oev/__init__.py", encoding="utf-8") as fh:
    init_src = fh.read()
init = re.search(r'__version__\s*=\s*"([^"]+)"', init_src).group(1)
if py == init:
    print(f"version parity OK: {py}")
else:
    fail.append(f"version mismatch: pyproject {py} != oev/__init__.py {init}")

with open("docs/claims.json", encoding="utf-8") as fh:
    claims = json.load(fh)["claims"]
known = {str(c["value"]) for c in claims}
# canonical short forms so 0.776 matches 0.7760, 0.93 matches 0.9300
known |= {v.rstrip("0").rstrip(".") if "." in v else v for v in known}

text = ""
for f in ["README.md", "BENCHMARKS.md"]:
    with open(f, encoding="utf-8") as fh:
        text += fh.read()
accs = set(re.findall(r"0\.\d{3,4}", text))
unknown = {a for a in accs
           if a not in known and a not in COMPETITOR and a not in EXEMPT
           and a.rstrip("0") not in known and a.rstrip("0") not in COMPETITOR}
if unknown:
    fail.append(f"OEV numbers in docs but not in claims.json: {sorted(unknown)}")
else:
    print(f"claims cross-check OK ({len(accs)} accuracy-like numbers)")

placeholders = [c for c in claims if "..." in c.get("eval", "")]
if placeholders:
    fail.append(f"placeholder eval commands in claims.json: {len(placeholders)}")
else:
    print(f"eval commands OK ({len(claims)} claims, no placeholders)")

if fail:
    print("\nFAIL:")
    for f in fail:
        print(" -", f)
    sys.exit(1)
print("\nall integrity checks passed")
