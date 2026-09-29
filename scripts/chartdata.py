"""Chart data: measured and published numbers behind the README figures.

Charted values must stay consistent with docs/claims.json (the published
number registry) and BENCHMARKS.md. load() cross-checks every value that has
a matching metric+benchmark claim there, so a BENCHMARKS edit can never
silently diverge from the charts. Entries tagged with a (checked_at, claim)
pair are verified against the registry at import time; chart-only numbers
(baselines, illustrative points, per-workflow rows that predate the registry)
live here with their provenance comments and are NOT in the registry.

Values are in magnitude-1 float (0.7705, not 77.05).
"""
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# =====================================================================
# OEV: the registered, receipt-backed numbers used by the charts
# =====================================================================

# (value, metric, benchmark, checkpoint-substring): each must match exactly one
# claim in docs/claims.json. This is the divergence guard.
CHECKED_OEV = [
    (0.7760, "accuracy", "typed-decisions", "ensemble"),
    (0.9489, "accuracy", "ag_news", "oev-tiny_bb"),
    (0.6505, "accuracy", "dair_emotion_zero_shot", "distill-round2b"),
    (0.8584, "accuracy", "banking77", "b77soup"),
    (0.0938, "ece", "typed-decisions", "oev-base-td5 (raw)"),
    (0.0279, "ece", "typed-decisions", "oev-base-rlcd"),
    (0.0204, "ece", "typed-decisions (temperature calibrated)", "temperature 0.5981"),
    (22.2, "latency_p50_ms", "typed-decisions", "oev-base-td5"),
    (15.9, "latency_p50_ms", "typed-decisions", "oev-base-td5"),
    (447, "cpu latency p50", "single-question inference", "b77soup"),
    (0.0298, "ece", "typed-decisions", "sharpened ensemble"),
    (0.7020, "soft_accuracy", "typed-decisions", "sharpened ensemble"),
    (0.735, "accuracy", "typed-decisions", "r2b-student+td5"),
    (0.7705, "accuracy", "typed-decisions", "oev-base-td5"),
]

# shared public benchmarks (in-domain). Kev publishes none of these.
# emotion = shipped student zero-shot like laya/Jev; 0.8584 = soup. The
# fine-tuned emotion specialist scores 0.9300 (BENCHMARKS.md table, not charted)
BENCH = ["typed-decisions", "AG News", "emotion (zero-shot)", "Banking77"]
OEVD = [0.7760, 0.9489, 0.6505, 0.8584]

# competitor numbers quoted from published tables (not OEV claims)
LAYA = [0.766, 0.950, 0.595, 0.425]
JEV = [0.727, 0.910, 0.480, 0.870]

# OEV T4 latency pair (single, batch-32 per-question); laya single 39.5 published
OEV_LAT = [22.2, 15.9]
OEV_LAT_BATCH_LABELS = ["1 question", "batch 32\n(per question)"]

# calibration panel in the big figure: laya and Jev ECE are quoted from their
# published tables; the OEV pair is checked above
ECE_PANEL = [0.213, 0.0938, 0.0298, 0.144]

# zero-shot / OOD transfer, one dot per model, attributed per label.
# WANLI = td5 0.3945, ANLI = round-1 student 0.3380 (per-checkpoint tables in
# BENCHMARKS.md); the r2b emotion row and the Kev columns are checked above
# where registered, quoted from published tables where not.
ZS_ROWS = ["emotion (zero-shot)", "WANLI OOD (zero-shot)", "ANLI R1 (zero-shot)", "Kev new sources (own tasks)"]
ZS_OEV = [0.6505, 0.3945, 0.3380, None]
ZS_LAYA = [0.595, None, None, None]
ZS_KEV = [None, None, None, 0.838]   # Kev-4B, test split of its new-sources protocol
ZS_WHO = [("OEV", ZS_OEV, None), ("laya", ZS_LAYA, None), ("Kev", ZS_KEV, None)]

# latency (published or measured); laya 32.8-39.5 and Jev 236-276 are ranges
# charted at the midpoint; Kev-4B 41.5 is L40S
LAT_LABELS = [
    "OEV\n(184M, T4)",
    "laya\n(range midpoint)",
    "Kev-4B\n(L40S)",
    "Jev\n(range midpoint)",
]
LAT_VALS = [22.2, 36.2, 41.5, 256.0]

# calibration: typed-decisions ECE, lower is better. Temperature fitted on
# the valid split only (receipt runs/20260928-092030-calibration-oev-base-td5.json)
CAL_LABELS = ["td5 raw", "previous best\n(rlcd-soup)", "td5 + temperature\n0.598 (new best)"]
CAL_VALS = [0.0938, 0.0279, 0.0204]
CAL_COLORS = ["#b0b8c0", "#e8d5a3", "#b5d4bf"]

# headline scorecard: single and ensemble typed accuracy (registered above)
HEAD_SCORE = [0.7705, 0.7760]

# scorecard latency bars: registered T4 p50 and the CPU laptop row (checked above)
SCORE_LAT = [22.2, 447]

# banking77 soup accuracy for the scorecard and size points (checked above)
B77_SOUP = 0.8584

# accuracy vs size (published points only; Jev size not published)
PTS = [
    ("laya (AG News)", 421, 0.950, "red"),
    ("laya (typed)", 421, 0.766, "red"),
    ("OEV base (AG News)", 184, 0.9489, "blue"),
    ("OEV base (emotion)", 184, 0.9300, "blue"),
    ("OEV soup (b77)", 184, 0.8584, "blue"),
    ("OEV ensemble", 4 * 184, 0.7760, "blue"),
    ("OEV single", 184, 0.7705, "blue"),
    ("Kev-0.8B (new sources)", 800, 0.697, "green"),
    ("Kev-4B (new sources)", 4000, 0.838, "green"),
    ("Kev-9B (new sources)", 9000, 0.852, "green"),
    ("Kev-27B (new sources)", 27000, 0.896, "green"),
]

# the big figure's size panel: same values as PTS, fewer points with its own
# labels and label offsets (the two scripts show different selections)
BIGFIG_PTS = [
    ("laya (421M)", 421, 0.766, "red"),
    ("OEV single (184M)", 184, 0.7705, "blue"),
    ("OEV ensemble (736M)", 736, 0.7760, "blue"),
    ("OEV b77 soup (184M)", 184, 0.8584, "blue"),
    ("Kev-0.8B (new sources)", 800, 0.697, "green"),
    ("Kev-4B (new sources)", 4000, 0.838, "green"),
    ("Kev-9B (new sources)", 9000, 0.852, "green"),
    ("Kev-27B (new sources)", 27000, 0.896, "green"),
]
BIGFIG_POFF = {"laya (421M)": (6, -12), "OEV single (184M)": (6, -12),
               "OEV ensemble (736M)": (6, 6), "OEV b77 soup (184M)": (-20, 8),
               "Kev-0.8B (new sources)": (0, 8), "Kev-4B (new sources)": (-40, 8),
               "Kev-9B (new sources)": (-10, 8), "Kev-27B (new sources)": (-60, 8)}

# per-workflow accuracy (typed-decisions). Chart-only: these per-workflow rows
# predate the registry; BENCHMARKS.md holds the full table they come from.
WF_LABELS = ["invoice\nprocessing", "customer\nservice", "agent-trace\nobservability", "security\nincidents"]
WF_LAYA = [0.804, 0.764, 0.730, 0.766]
WF_OEV = [0.8360, 0.8040, 0.7400, 0.7220]

# typed ensemble sharpening sweep (gamma). Chart-only: the 2.5 endpoint is
# registered as exploratory and the intermediate gammas come from that sweep.
GAMMA_SOFT = [0.5871, 0.6410, 0.6758, 0.7020]
GAMMA_HARD = [0.7755, 0.7750, 0.7750, 0.7730]

# typed-decisions vs baselines. Chart-only: teacher ceiling is the published
# teacher ensemble mean, floors are protocol constants.
BASELINE_NAMES = ["random\nguess", "majority\nclass", "teacher\nceiling", "Jev", "laya-typed\ndecisions", "OEV\nensemble"]
BASELINE_VALS = [0.318, 0.461, 0.735, 0.727, 0.766, 0.7760]


def _fmt(value):
    # match the registry's canonical short forms (0.7705 -> "0.7705", 0.93 -> "0.93")
    s = str(value)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def load():
    # cross-check registered chart numbers against docs/claims.json. Fails
    # loudly when BENCHMARKS/claims and the charts would diverge.
    claims = json.load(open(_REPO / "docs" / "claims.json", encoding="utf-8"))["claims"]
    indexed = {}
    for c in claims:
        key = (c["metric"], c["benchmark"], _fmt(c["value"]))
        indexed.setdefault(key, []).append(c.get("checkpoint", ""))
    problems = []
    for value, metric, benchmark, needle in CHECKED_OEV:
        # benchmark may be a prefix: the registry spells some benchmarks out
        # ("single-question inference, 3 states, 30 calls, warmup excluded")
        hits = [c for (m, b, v), cs in indexed.items() if m == metric and b.startswith(benchmark)
                for c in cs if (m, b, v) == (metric, b, _fmt(value))]
        if not hits:
            problems.append(f"{metric}/{benchmark}={value}: no claim found")
            continue
        if not any(needle in c for c in hits):
            problems.append(f"{metric}/{benchmark}={value}: claim exists but checkpoint '{needle}' not matched")
    if problems:
        raise SystemExit("chart data diverged from docs/claims.json:\n  " + "\n  ".join(problems))
    return {"n_claims": len(claims)}


def oev_color(name):
    # map the neutral names in PTS to palette entries (kept separate so the
    # data module stays importable without a plot backend)
    import chartstyle

    return {"blue": chartstyle.P_BLUE, "red": chartstyle.P_RED, "green": chartstyle.P_GREEN}[name]
