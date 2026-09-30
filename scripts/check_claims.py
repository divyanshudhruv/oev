"""Claims integrity checks: python scripts/check_claims.py

Fails on version mismatch, OEV numbers in docs missing from
docs/claims.json, headline claims whose value diverges between
README, BENCHMARKS and the registry (semantic pins), or placeholder
eval commands. Competitor numbers are quoted from published tables,
not claims.
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

# headline numbers that must resolve to the same registered claim everywhere
# they appear. a number here must (a) exist as an exact claim of that metric
# and benchmark in claims.json and (b) appear in both README.md and
# BENCHMARKS.md - edit claims.json and both docs together, or drop the pin.
PINNED = {
    "0.7705": ("accuracy", "typed-decisions", 0.7705),
    "0.7760": ("accuracy", "typed-decisions", 0.776),
    "0.8594": ("accuracy", "banking77", 0.8594),
    "0.0583": ("ece", "banking77", 0.0583),
    "0.8700": ("accuracy", "dair_emotion_zero_shot", 0.87),
    "0.5750": ("accuracy", "anli_r1_finetuned", 0.575),
    "22.2": ("latency_p50_ms", "typed-decisions", 22.2),
}

texts = {}
for f in ["README.md", "BENCHMARKS.md"]:
    with open(f, encoding="utf-8") as fh:
        texts[f] = fh.read()
for num, (metric, bench, value) in PINNED.items():
    short = num.rstrip("0").rstrip(".") if "." in num else num
    if not any(str(c["value"]) in (num, short) and c["metric"] == metric
               and c["benchmark"].startswith(bench) for c in claims):
        fail.append(f"pinned {num}: no matching {metric}/{bench} claim in claims.json")
    for f, body in texts.items():
        if num not in body and f".{short.lstrip('0')}" not in body:
            fail.append(f"pinned {num}: missing from {f}")
if not [x for x in fail if x.startswith("pinned")]:
    print(f"semantic anchors OK ({len(PINNED)} headline numbers consistent across docs)")

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
