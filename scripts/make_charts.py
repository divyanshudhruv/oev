"""Generate README charts from measured OEV numbers into assets/.

Every number below is a measured or published figure:
- OEV: BENCHMARKS.md / docs/claims.json (this repo, measured)
- laya: published by NandhaKishorM/laya
- Jev: published by TypeSafe
- Kev: published by jaredpalmer/kev (OOD-suite accuracies; no comparable
  in-domain numbers are published for typed/AG News/emotion/b77)

Light + dark variants are written for every chart.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("assets", exist_ok=True)

ACC = ["#4361ee", "#e63946", "#2a9d8f", "#e9c46a", "#8d99ae"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#e8e8e8",
    "grid.linewidth": 0.5,
    "axes.titlepad": 14,
    "figure.autolayout": False,
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# =====================================================================
# measured / published numbers (single source of truth for this script)
# =====================================================================

# shared public benchmarks (in-domain). Kev publishes none of these.
BENCH = ["typed-decisions", "AG News", "emotion", "Banking77"]
OEVD = [0.7760, 0.9489, 0.9300, 0.8584]   # 0.8584 = soup, beats the 0.8529 ensemble
LAYA = [0.766, 0.950, 0.595, 0.425]
JEV = [0.727, 0.910, 0.480, 0.870]

# zero-shot / OOD transfer (each model never trained on the dataset).
# OEV: shipped round-2b distilled student on emotion (0.6505, beats laya's zero-shot
# head-to-head; round-1 peaked 0.6875), wanli/anli at the NLI floor, measured 2026-09-25.
# laya emotion 0.595 is published as zero-shot. Kev publishes OOD-suite numbers on its
# own suite. ANLI measured at chance (0.326).
ZS = ["emotion\n(zero-shot)", "WANLI OOD\n(zero-shot)", "ANLI R1\n(zero-shot)", "Kev OOD suite\n(own tasks)"]
OEVD_ZS = [0.6505, 0.3450, 0.3260, None]
LAYA_ZS = [0.595, None, None, None]
JEV_ZS = [None, None, None, None]
KEV_ZS = [None, None, None, 0.845]   # midpoint of Kev's published 0.837-0.852 range

# latency (published or measured)
LAT_LABELS = ["OEV\n(184M, T4)", "laya\n(421M, T4)", "Kev-4B\n(L40S)", "Jev\n(API RT)"]
LAT_VALS = [22.2, 36.2, 36.0, 236.0]   # laya 32.8-39.5 midpoint; Kev 36ms on L40S

# =====================================================================
# chart 1: shared in-domain benchmarks (OEV vs laya vs Jev)
# =====================================================================

fig, ax = plt.subplots(figsize=(12, 6.2))
x = range(len(BENCH))
w = 0.26
b1 = ax.bar([i - w for i in x], JEV, w, label="Jev (closed API, published)", color=ACC[4])
b2 = ax.bar(list(x), LAYA, w, label="laya (published)", color=ACC[1])
b3 = ax.bar([i + w for i in x], OEVD, w, label="OEV (this repo, soup/ensemble)", color=ACC[0])
ax.set_xticks(list(x))
ax.set_xticklabels(BENCH)
ax.set_ylim(0, 1.18)
ax.set_ylabel("accuracy")
ax.set_title("OEV vs published numbers on shared public benchmarks", fontweight="bold", pad=15)
ax.legend(frameon=False, loc="upper left", fontsize=12)
for b in (b1, b2, b3):
    ax.bar_label(b, fmt="%.3f", fontsize=10, padding=3)
fig.tight_layout()
fig.savefig("assets/benchmarks.png", dpi=150)
plt.close(fig)

# =====================================================================
# chart 2: zero-shot / OOD transfer (the honest row)
# =====================================================================

fig, ax = plt.subplots(figsize=(13, 5.6))
xz = range(len(ZS))
wz = 0.2
series = [("OEV (distilled student, measured)", OEVD_ZS, ACC[0]),
          ("laya (published)", LAYA_ZS, ACC[1]),
          ("Kev (published, own suite)", KEV_ZS, ACC[2])]
for si, (label, vals, color) in enumerate(series):
    offs = [i + (si - 1) * wz for i in xz]
    vals_plot = [v if v is not None else 0 for v in vals]
    bars = ax.bar(offs, vals_plot, wz, label=label, color=color)
    for rect, v in zip(bars, vals):
        if v is None:
            rect.set_alpha(0.12)
    ax.bar_label(bars, labels=[f"{v:.3f}" if v is not None else "not published" for v in vals],
                 fontsize=9, padding=2)
    ax.bar_label(bars, labels=["" for _ in vals])  # keep default formatting off
ax.axhline(0.333, color="#888", ls=":", lw=1.2)
ax.annotate("random floor (3 classes)", (3.32, 0.345), fontsize=8, color="#666")
ax.set_xticks(list(xz))
ax.set_xticklabels(ZS, fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_ylabel("accuracy")
ax.set_title("zero-shot / out-of-domain transfer (faded bar = not published)", fontweight="bold", pad=15)
ax.legend(frameon=False, loc="upper right", fontsize=10)
fig.tight_layout()
fig.savefig("assets/zeroshot.png", dpi=150)
plt.close(fig)

# =====================================================================
# chart 2b: transfer + latency combined (one figure, two panels)
# =====================================================================

fig, (axt, axl) = plt.subplots(1, 2, figsize=(15.5, 5.4), gridspec_kw={"width_ratios": [1.45, 1]})

# left panel: zero-shot / OOD transfer
for si, (label, vals, color) in enumerate(series):
    offs = [i + (si - 1) * wz for i in xz]
    vals_plot = [v if v is not None else 0 for v in vals]
    bars = axt.bar(offs, vals_plot, wz, label=label, color=color)
    for rect, v in zip(bars, vals):
        if v is None:
            rect.set_alpha(0.12)
    axt.bar_label(bars, labels=[f"{v:.3f}" if v is not None else "" for v in vals],
                  fontsize=8.5, padding=2)
axt.axhline(0.333, color="#888", ls=":", lw=1.2)
axt.annotate("random floor (3 classes)", (3.3, 0.35), fontsize=8, color="#666")
axt.set_xticks(list(xz))
axt.set_xticklabels(ZS, fontsize=9)
axt.set_ylim(0, 1.08)
axt.set_ylabel("accuracy")
axt.set_title("zero-shot / OOD transfer", fontweight="bold", pad=12, fontsize=13)

# right panel: latency
lcolors = [ACC[0], ACC[1], ACC[2], ACC[4]]
lbars = axl.bar([i for i in range(len(LAT_LABELS))], LAT_VALS, color=lcolors)
axl.set_xticks(range(len(LAT_LABELS)))
axl.set_xticklabels([l.replace("\n", " ") for l in LAT_LABELS], fontsize=9)
axl.bar_label(lbars, fmt="%.1f ms", fontsize=9, padding=3)
axl.set_ylim(0, 270)
axl.set_ylabel("ms per question")
axl.set_title("latency (lower is better)", fontweight="bold", pad=12, fontsize=13)

# legend belongs to the left panel: anchor it just above that axes only
axt.legend(ncol=3, frameon=False, fontsize=10, loc="lower left",
           bbox_to_anchor=(0, 1.01), columnspacing=1.4, handlelength=1.6)
fig.tight_layout()
fig.savefig("assets/transfer_speed.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# =====================================================================
# chart 3: accuracy vs params (all published points)
# =====================================================================

PTS = [
    ("laya-typed-decisions", 421, 0.766, ACC[1]),
    ("laya (AG News)", 421, 0.950, ACC[1]),
    ("OEV ensemble", 4 * 184, 0.7760, ACC[0]),
    ("OEV single", 184, 0.7705, ACC[0]),
    ("OEV soup (b77)", 184, 0.8584, ACC[0]),
    ("OEV base (AG News)", 184, 0.9489, ACC[0]),
    ("OEV base (emotion)", 184, 0.9300, ACC[0]),
    ("Jev (API, size n/a)", 700, 0.727, ACC[4]),
    ("Kev-0.8B (OOD suite)", 800, 0.837, ACC[2]),
    ("Kev-4B (OOD suite)", 4000, 0.852, ACC[2]),
]

fig, ax = plt.subplots(figsize=(7.5, 4.8))
for name, params, acc, c in PTS:
    ax.scatter(params, acc, s=130, color=c, zorder=3, edgecolors="black", linewidths=0.5)
    dx, dy = (6, -12) if "Jev" in name else (6, 4)
    ax.annotate(name, (params, acc), textcoords="offset points", xytext=(dx, dy), fontsize=9)
ax.set_xlabel("parameters (millions)")
ax.set_ylabel("accuracy")
ax.set_title("accuracy vs model size", fontweight="bold")
ax.set_xlim(0, 4200)
fig.tight_layout()
fig.savefig("assets/params_vs_acc.png", dpi=150)
plt.close(fig)

# =====================================================================
# chart 4: sharpening sweep (soft acc vs gamma)
# =====================================================================

fig, ax = plt.subplots(figsize=(7.5, 4.5))
gammas = [1.0, 1.5, 2.0, 2.5]
soft = [0.5871, 0.6410, 0.6758, 0.7020]
hard = [0.7755, 0.7750, 0.7750, 0.7730]
ax.plot(gammas, soft, "o-", color=ACC[0], lw=2, label="soft accuracy")
ax.plot(gammas, hard, "s--", color=ACC[2], lw=2, label="hard accuracy")
ax.axhline(0.471, color=ACC[1], ls=":", lw=1.5)
ax.annotate("laya soft acc 0.471", (1.05, 0.477), color=ACC[1], fontsize=9)
for g, s in zip(gammas, soft):
    ax.annotate(f"{s:.4f}", (g, s), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
ax.set_xlabel("sharpening exponent gamma")
ax.set_ylabel("accuracy")
ax.set_title("post-hoc sharpening sweep (3-voter ensemble)", fontweight="bold")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("assets/sharpening.png", dpi=150)
plt.close(fig)

# =====================================================================
# chart 5: per-workflow accuracy (typed-decisions)
# =====================================================================

fig, ax = plt.subplots(figsize=(9, 4.8))
workflows = ["invoice\nprocessing", "customer\nservice", "agent-trace\nobservability", "security\nincidents"]
laya_wf = [0.804, 0.764, 0.730, 0.766]
oev_wf = [0.8360, 0.8040, 0.7400, 0.7220]
xw = range(len(workflows))
ww = 0.34
b1 = ax.bar([i - ww / 2 for i in xw], laya_wf, ww, label="laya (published)", color=ACC[1])
b2 = ax.bar([i + ww / 2 for i in xw], oev_wf, ww, label="OEV (4-voter ensemble)", color=ACC[0])
ax.bar_label(b1, fmt="%.3f", fontsize=9, padding=2)
ax.bar_label(b2, fmt="%.3f", fontsize=9, padding=2)
ax.axhline(0.766, color=ACC[1], ls=":", lw=1)
ax.axhline(0.776, color=ACC[0], ls=":", lw=1)
ax.set_xticks(list(xw))
ax.set_xticklabels(workflows)
ax.set_ylim(0.6, 0.9)
ax.set_ylabel("accuracy")
ax.set_title("typed-decisions per-workflow: OEV wins 3 of 4", fontweight="bold")
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig("assets/workflows.png", dpi=150)
plt.close(fig)

# =====================================================================
# chart 6: latency (published / measured)
# =====================================================================

fig, ax = plt.subplots(figsize=(8.5, 4.6))
colors = [ACC[0], ACC[1], ACC[2], ACC[4]]
bars = ax.bar(LAT_LABELS, LAT_VALS, color=colors)
ax.bar_label(bars, fmt="%.1f ms", fontsize=10, padding=3)
ax.set_ylabel("ms per question")
ax.set_title("latency: lower is better (each model's best published hardware)", fontweight="bold")
ax.set_ylim(0, 270)
fig.tight_layout()
fig.savefig("assets/latency.png", dpi=150)
plt.close(fig)

print("light charts written: benchmarks, zeroshot, params_vs_acc, sharpening, workflows, latency")

# =====================================================================
# dark variants (reset rcParams first so nothing inherits light values)
# =====================================================================

DARK_BG = "#0d1117"
DARK_GRID = "#3d444d"
DARK_TEXT = "#e6edf3"
DARK_ACC = ["#79b8ff", "#ff7b72", "#56d364", "#e3b341", "#a3a3a3"]

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": DARK_BG,
    "axes.grid": True,
    "grid.color": DARK_GRID,
    "text.color": DARK_TEXT,
    "axes.edgecolor": DARK_GRID,
    "axes.labelcolor": DARK_TEXT,
    "xtick.color": DARK_TEXT,
    "ytick.color": DARK_TEXT,
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def _dark_bars(ax, bars):
    for b in bars:
        b.set_edgecolor(DARK_TEXT)
        b.set_linewidth(0.4)

# benchmarks (dark)
fig, ax = plt.subplots(figsize=(12, 6.2))
b1 = ax.bar([i - w for i in x], JEV, w, label="Jev (closed API, published)", color=DARK_ACC[4])
b2 = ax.bar(list(x), LAYA, w, label="laya (published)", color=DARK_ACC[1])
b3 = ax.bar([i + w for i in x], OEVD, w, label="OEV (this repo, soup/ensemble)", color=DARK_ACC[0])
_dark_bars(ax, b1 + b2 + b3)
ax.set_xticks(list(x)); ax.set_xticklabels(BENCH)
ax.set_ylim(0, 1.18); ax.set_ylabel("accuracy")
ax.set_title("OEV vs published numbers on shared public benchmarks", fontweight="bold", pad=15)
ax.legend(frameon=False, loc="upper left", fontsize=12)
for b in (b1, b2, b3):
    ax.bar_label(b, fmt="%.3f", fontsize=10, padding=3, color=DARK_TEXT)
fig.tight_layout(); fig.savefig("assets/benchmarks_dark.png", dpi=150); plt.close(fig)

# zero-shot (dark)
fig, ax = plt.subplots(figsize=(13, 5.6))
for si, (label, vals, color) in enumerate(series):
    offs = [i + (si - 1) * wz for i in xz]
    vals_plot = [v if v is not None else 0 for v in vals]
    bars = ax.bar(offs, vals_plot, wz, label=label, color=color)
    for rect, v in zip(bars, vals):
        if v is None:
            rect.set_alpha(0.12)
    ax.bar_label(bars, labels=[f"{v:.3f}" if v is not None else "not published" for v in vals],
                 fontsize=9, padding=2, color=DARK_TEXT)
ax.axhline(0.333, color=DARK_TEXT, ls=":", lw=1.2, alpha=0.5)
ax.annotate("random floor (3 classes)", (3.32, 0.345), fontsize=8, color=DARK_TEXT, alpha=0.7)
ax.set_xticks(list(xz)); ax.set_xticklabels(ZS, fontsize=10)
ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy")
ax.set_title("zero-shot / out-of-domain transfer (faded bar = not published)", fontweight="bold", pad=15)
ax.legend(frameon=False, loc="upper right", fontsize=10)
fig.tight_layout(); fig.savefig("assets/zeroshot_dark.png", dpi=150); plt.close(fig)

# transfer + latency combined (dark)
fig, (axt, axl) = plt.subplots(1, 2, figsize=(15.5, 5.4), gridspec_kw={"width_ratios": [1.45, 1]})
for si, (label, vals, color) in enumerate(series):
    offs = [i + (si - 1) * wz for i in xz]
    vals_plot = [v if v is not None else 0 for v in vals]
    bars = axt.bar(offs, vals_plot, wz, label=label, color=color)
    for rect, v in zip(bars, vals):
        if v is None:
            rect.set_alpha(0.12)
    axt.bar_label(bars, labels=[f"{v:.3f}" if v is not None else "" for v in vals],
                  fontsize=8.5, padding=2, color=DARK_TEXT)
axt.axhline(0.333, color=DARK_TEXT, ls=":", lw=1.2, alpha=0.5)
axt.annotate("random floor (3 classes)", (3.3, 0.35), fontsize=8, color=DARK_TEXT, alpha=0.7)
axt.set_xticks(list(xz)); axt.set_xticklabels(ZS, fontsize=9)
axt.set_ylim(0, 1.08); axt.set_ylabel("accuracy")
axt.set_title("zero-shot / OOD transfer", fontweight="bold", pad=12, fontsize=13)
lcolors = [DARK_ACC[0], DARK_ACC[1], DARK_ACC[2], DARK_ACC[4]]
lbars = axl.bar(range(len(LAT_LABELS)), LAT_VALS, color=lcolors)
axl.set_xticks(range(len(LAT_LABELS)))
axl.set_xticklabels([l.replace("\n", " ") for l in LAT_LABELS], fontsize=9)
_dark_bars(axl, lbars)
axl.bar_label(lbars, fmt="%.1f ms", fontsize=9, padding=3, color=DARK_TEXT)
axl.set_ylim(0, 270); axl.set_ylabel("ms per question")
axl.set_title("latency (lower is better)", fontweight="bold", pad=12, fontsize=13)
axt.legend(ncol=3, frameon=False, fontsize=10, loc="lower left",
           bbox_to_anchor=(0, 1.01), columnspacing=1.4, handlelength=1.6,
           labelcolor=DARK_TEXT)
fig.tight_layout()
fig.savefig("assets/transfer_speed_dark.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# params vs acc (dark)
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for name, params, acc, _ in PTS:
    c = DARK_ACC[0] if name.startswith("OEV") else (DARK_ACC[1] if "laya" in name
        else (DARK_ACC[2] if "Kev" in name else DARK_ACC[4]))
    ax.scatter(params, acc, s=130, color=c, zorder=3, edgecolors=DARK_TEXT, linewidths=0.5)
    dx, dy = (6, -12) if "Jev" in name else (6, 4)
    ax.annotate(name, (params, acc), textcoords="offset points", xytext=(dx, dy), fontsize=9, color=DARK_TEXT)
ax.set_xlabel("parameters (millions)"); ax.set_ylabel("accuracy")
ax.set_title("accuracy vs model size", fontweight="bold"); ax.set_xlim(0, 4200)
fig.tight_layout(); fig.savefig("assets/params_vs_acc_dark.png", dpi=150); plt.close(fig)

# sharpening (dark)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(gammas, soft, "o-", color=DARK_ACC[0], lw=2, label="soft accuracy")
ax.plot(gammas, hard, "s--", color=DARK_ACC[2], lw=2, label="hard accuracy")
ax.axhline(0.471, color=DARK_ACC[1], ls=":", lw=1.5)
ax.annotate("laya soft acc 0.471", (1.05, 0.477), color=DARK_ACC[1], fontsize=9)
for g, sv in zip(gammas, soft):
    ax.annotate(f"{sv:.4f}", (g, sv), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center", color=DARK_TEXT)
ax.set_xlabel("sharpening exponent gamma"); ax.set_ylabel("accuracy")
ax.set_title("post-hoc sharpening sweep (3-voter ensemble)", fontweight="bold")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("assets/sharpening_dark.png", dpi=150); plt.close(fig)

# workflows (dark)
fig, ax = plt.subplots(figsize=(9, 4.8))
b1 = ax.bar([i - ww / 2 for i in xw], laya_wf, ww, label="laya (published)", color=DARK_ACC[1])
b2 = ax.bar([i + ww / 2 for i in xw], oev_wf, ww, label="OEV (4-voter ensemble)", color=DARK_ACC[0])
_dark_bars(ax, b1 + b2)
ax.bar_label(b1, fmt="%.3f", fontsize=9, padding=2, color=DARK_TEXT)
ax.bar_label(b2, fmt="%.3f", fontsize=9, padding=2, color=DARK_TEXT)
ax.axhline(0.766, color=DARK_ACC[1], ls=":", lw=1)
ax.axhline(0.776, color=DARK_ACC[0], ls=":", lw=1)
ax.set_xticks(list(xw)); ax.set_xticklabels(workflows)
ax.set_ylim(0.6, 0.9); ax.set_ylabel("accuracy")
ax.set_title("typed-decisions per-workflow: OEV wins 3 of 4", fontweight="bold")
ax.legend(frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig("assets/workflows_dark.png", dpi=150); plt.close(fig)

# latency (dark)
fig, ax = plt.subplots(figsize=(8.5, 4.6))
colors = [DARK_ACC[0], DARK_ACC[1], DARK_ACC[2], DARK_ACC[4]]
bars = ax.bar(LAT_LABELS, LAT_VALS, color=colors)
_dark_bars(ax, bars)
ax.bar_label(bars, fmt="%.1f ms", fontsize=10, padding=3, color=DARK_TEXT)
ax.set_ylabel("ms per question")
ax.set_title("latency: lower is better (each model's best published hardware)", fontweight="bold")
ax.set_ylim(0, 270)
fig.tight_layout(); fig.savefig("assets/latency_dark.png", dpi=150); plt.close(fig)

print("dark variants written")
