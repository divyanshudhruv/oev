"""Generate README charts from measured OEV numbers into assets/.

Every number below is a measured or published figure:
- OEV: BENCHMARKS.md (this repo, measured)
- laya: published by NandhaKishorM/laya
- Jev: published by TypeSafe
- Kev: published by jaredpalmer/kev (OOD-suite accuracies; its in-domain
  rows are omitted where it publishes no comparable number)

Charts use transparent backgrounds so the surrounding page shows through.
Light variants use dark text; dark variants use light text. Colors are
pastel so nothing vibrates against either background.

Light + dark variants are written for every chart.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

os.makedirs("assets", exist_ok=True)

# pastel palette: legible on both light and dark pages
P_BLUE, P_RED, P_GREEN, P_SAND, P_GRAY = "#a8c5e6", "#f0a8a8", "#b5d4bf", "#e8d5a3", "#b0b8c0"
ACC = [P_BLUE, P_RED, P_GREEN, P_SAND, P_GRAY]

plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.transparent": True,
    "axes.grid": True,
    "grid.color": "#c8ccd0",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.6,
    "axes.titlepad": 14,
    "figure.autolayout": False,
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": "#24292f",
    "axes.edgecolor": "#57606a",
    "axes.labelcolor": "#24292f",
    "xtick.color": "#24292f",
    "ytick.color": "#24292f",
})

# =====================================================================
# measured / published numbers (single source of truth for this script)
# =====================================================================

# shared public benchmarks (in-domain). Kev publishes none of these.
BENCH = ["typed-decisions", "AG News", "emotion", "Banking77"]
OEVD = [0.7760, 0.9489, 0.9300, 0.8584]   # 0.8584 = soup, beats the 0.8529 ensemble
LAYA = [0.766, 0.950, 0.595, 0.425]
JEV = [0.727, 0.910, 0.480, 0.870]

# zero-shot / OOD transfer, one dot per model, attributed per label.
# OEV dots: emotion = r2b distilled student 0.6505 (beats laya's zero-shot
# head-to-head); WANLI = td5 0.3945; ANLI = round-1 student 0.3380.
# BENCHMARKS.md holds the per-checkpoint tables.
ZS_ROWS = ["emotion (zero-shot)", "WANLI OOD (zero-shot)", "ANLI R1 (zero-shot)", "Kev OOD suite (own tasks)"]
ZS_OEV = [0.6505, 0.3945, 0.3380, None]
ZS_LAYA = [0.595, None, None, None]
ZS_KEV = [None, None, None, 0.845]   # published range 0.837-0.852; plotted midpoint
ZS_WHO = [("OEV", ZS_OEV, P_BLUE), ("laya", ZS_LAYA, P_RED), ("Kev", ZS_KEV, P_GREEN)]

# latency (published or measured); ranges use midpoint
LAT_LABELS = [
    "OEV\n(184M, T4)",
    "laya\n(range midpoint)",
    "Kev-4B\n(L40S)",
    "Jev\n(range midpoint)",
]
LAT_VALS = [22.2, 36.2, 36.0, 256.0]   # laya 32.8-39.5 and Jev 236-276 are ranges

# checkpoint x benchmark matrix (accuracy; None = not measured).
# rows: td5 (typed specialist), mt (generalist teacher), R1 student,
# R2 student (collapsed), R2b student (shipped), MNLI specialist.
MX_ROWS = ["td5", "mt", "R1 student", "R2 student", "R2b student", "MNLI spec"]
MX_COLS = ["typed", "Banking77", "emotion", "AG News", "WANLI", "ANLI"]
MX = [
    [0.7705, None,   0.4265, 0.9489, 0.3945, 0.3360],
    [0.7350, None,   0.5960, 0.9489, 0.3800, 0.3360],
    [0.5385, 0.8205, 0.6875, None,   0.3450, 0.3380],
    [0.1917, 0.0104, 0.2905, None,   0.3450, 0.3360],
    [0.6480, 0.7964, 0.6505, 0.7983, 0.3450, 0.3260],
    [None,   None,   None,   None,   0.5645, 0.3360],
]
# chance floors per column, annotated under the heatmap
MX_FLOOR = ["floor 0.32", "floor 0.01", "floor 0.17", "floor 0.25", "floor 0.33", "floor 0.33"]

# Banking77 progression (this repo, measured)
B77_STEPS = ["first\nrelease", "re-\nmeasured", "re-tune\n(r2b init)", "re-tune\n+ TTA", "3-ckpt\nensemble", "soup\n(single file)"]
B77_VALS = [0.8303, 0.8302, 0.8370, 0.8383, 0.8529, 0.8584]
B77_JEV = 0.870

# r2b gamma sweep on the mix validation split (accuracy flat, ECE U-shaped)
GAMMAS = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
G_ECE = [0.6170, 0.3278, 0.1443, 0.0398, 0.0424, 0.0765]
G_ACC = [0.8450] * len(GAMMAS)
G_PICK = 1.2

# accuracy vs size (published points only; Jev size not published)
PTS = [
    ("laya (AG News)", 421, 0.950, P_RED),
    ("laya (typed)", 421, 0.766, P_RED),
    ("OEV base (AG News)", 184, 0.9489, P_BLUE),
    ("OEV base (emotion)", 184, 0.9300, P_BLUE),
    ("OEV soup (b77)", 184, 0.8584, P_BLUE),
    ("OEV ensemble", 4 * 184, 0.7760, P_BLUE),
    ("OEV single", 184, 0.7705, P_BLUE),
    ("Kev-0.8B (OOD suite)", 800, 0.837, P_GREEN),
    ("Kev-4B (OOD suite)", 4000, 0.852, P_GREEN),
]
# staggered annotation offsets so nearby 184M labels never collide
OFFS = {
    "OEV base (AG News)": (-10, 10),
    "OEV base (emotion)": (8, 6),
    "OEV soup (b77)": (-6, -16),
    "OEV ensemble": (6, 8),
    "OEV single": (6, -14),
    "laya (AG News)": (-10, 10),
    "laya (typed)": (8, -14),
    "Kev-0.8B (OOD suite)": (0, 10),
    "Kev-4B (OOD suite)": (-30, 10),
}

# per-workflow accuracy (typed-decisions)
WF_LABELS = ["invoice\nprocessing", "customer\nservice", "agent-trace\nobservability", "security\nincidents"]
WF_LAYA = [0.804, 0.764, 0.730, 0.766]
WF_OEV = [0.8360, 0.8040, 0.7400, 0.7220]


def _dot_legend(edge, size=9):
    return [Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                   markeredgecolor=edge, markersize=size, label=who)
            for who, _, c in ZS_WHO]


# =====================================================================
# chart 1: shared in-domain benchmarks (OEV vs laya vs Jev)
# =====================================================================

fig, ax = plt.subplots(figsize=(12, 6.2))
x = np.arange(len(BENCH))
w = 0.26
b1 = ax.bar(x - w, JEV, w, label="Jev (closed API, published)", color=ACC[4])
b2 = ax.bar(x, LAYA, w, label="laya (published)", color=ACC[1])
b3 = ax.bar(x + w, OEVD, w, label="OEV (this repo, soup/ensemble)", color=ACC[0])
ax.set_xticks(x)
ax.set_xticklabels(BENCH)
ax.set_ylim(0, 1.18)
ax.set_ylabel("accuracy")
ax.set_title("OEV vs published numbers on shared public benchmarks (Jev Banking77: 72 labels)",
             fontweight="bold", pad=15)
ax.legend(frameon=False, loc="upper left", fontsize=12)
for b in (b1, b2, b3):
    ax.bar_label(b, fmt="%.3f", fontsize=10, padding=3)
fig.tight_layout()
fig.savefig("assets/benchmarks.png", dpi=150, transparent=True)
plt.close(fig)

def _zs_annotate(ax, i, v, who, color, size=9.5):
    """Label a zero-shot dot; spread labels horizontally when dots sit close."""
    row = [(u, val) for u, vals, _ in ZS_WHO if (val := vals[i]) is not None]
    row.sort(key=lambda t: t[1])
    near = [u for u, val in row if u != who and abs(val - v) < 0.1]
    if near:
        # left dot label goes up-left, right dot label goes up-right
        leftmost = v == min(val for _, val in row)
        dx, ha = (-8, "right") if leftmost else (8, "left")
    else:
        dx, ha = 0, "center"
    ax.annotate(f"{who} {v:.3f}", (v, i), textcoords="offset points",
                xytext=(dx, 12), ha=ha, fontsize=size, color=color)


# =====================================================================
# chart 2: zero-shot / OOD transfer as dumbbells. Replaces the congested
# 12-bar grouped chart: one row per benchmark, one dot per model.
# =====================================================================

fig, ax = plt.subplots(figsize=(11, 4.6))
for i in range(len(ZS_ROWS)):
    pts = [(vals[i], who) for who, vals, _ in ZS_WHO if vals[i] is not None]
    if pts:
        ax.hlines(i, min(v for v, _ in pts), max(v for v, _ in pts),
                  color="#c8ccd0", lw=2, zorder=1)
    else:
        ax.annotate("all three: not published", (0.995, i),
                    xycoords=("axes fraction", "data"), ha="right", va="center",
                    fontsize=9, color="#8b949e", style="italic")
    for v, who in pts:
        ax.scatter(v, i, s=170, color=dict((w, c) for w, _, c in ZS_WHO)[who],
                   zorder=3, edgecolors="#57606a", linewidths=0.7)
        _zs_annotate(ax, i, v, who, None)
ax.axvline(0.333, color="#8b949e", ls=":", lw=1.2)
ax.annotate("random floor, 3 classes", (0.335, 3.45), fontsize=8.5, color="#8b949e")
ax.set_yticks(range(len(ZS_ROWS)))
ax.set_yticklabels(ZS_ROWS, fontsize=11)
ax.set_xlim(0, 1.0)
ax.set_ylim(3.7, -0.7)
ax.invert_yaxis()
ax.set_xlabel("accuracy")
ax.set_title("zero-shot / out-of-domain transfer (dot = measured, gap = not published)",
             fontweight="bold", pad=12)
fig.tight_layout()
fig.savefig("assets/zeroshot.png", dpi=150, transparent=True)
plt.close(fig)

# =====================================================================
# chart 2b: transfer + latency combined (one figure, two panels)
# =====================================================================

fig, (axt, axl) = plt.subplots(1, 2, figsize=(15.5, 5.2),
                               gridspec_kw={"width_ratios": [1.45, 1]})
for i in range(len(ZS_ROWS)):
    pts = [(vals[i], who) for who, vals, _ in ZS_WHO if vals[i] is not None]
    if pts:
        axt.hlines(i, min(v for v, _ in pts), max(v for v, _ in pts),
                   color="#c8ccd0", lw=2, zorder=1)
    for v, who in pts:
        axt.scatter(v, i, s=140, color=dict((w, c) for w, _, c in ZS_WHO)[who],
                    zorder=3, edgecolors="#57606a", linewidths=0.7)
        axt.annotate(f"{v:.3f}", (v, i), textcoords="offset points", xytext=(0, 11),
                     ha="center", fontsize=9)
axt.axvline(0.333, color="#8b949e", ls=":", lw=1.2)
axt.annotate("floor 0.333", (0.34, 3.45), fontsize=8.5, color="#8b949e")
axt.set_yticks(range(len(ZS_ROWS)))
axt.set_yticklabels(ZS_ROWS, fontsize=9.5)
axt.set_xlim(0, 1.0)
axt.set_ylim(3.7, -0.7)
axt.invert_yaxis()
axt.set_xlabel("accuracy")
axt.set_title("zero-shot / OOD transfer", fontweight="bold", pad=12, fontsize=13)
axt.legend(handles=_dot_legend("#57606a"), frameon=False, fontsize=9.5,
           loc="lower left", bbox_to_anchor=(0, 1.005), ncols=3,
           columnspacing=1.2, handletextpad=0.2)

lbars = axl.bar(range(len(LAT_LABELS)), LAT_VALS,
                color=[ACC[0], ACC[1], ACC[2], ACC[4]])
axl.set_xticks(range(len(LAT_LABELS)))
axl.set_xticklabels([l.replace("\n", " ") for l in LAT_LABELS], fontsize=9)
axl.bar_label(lbars, fmt="%.1f ms", fontsize=9, padding=3)
axl.set_ylim(0, 290)
axl.set_ylabel("ms per question")
axl.set_title("latency (lower is better; ranges use midpoint)", fontweight="bold",
              pad=12, fontsize=13)
fig.tight_layout()
fig.savefig("assets/transfer_speed.png", dpi=150, transparent=True)
plt.close(fig)

# =====================================================================
# chart 3: checkpoint x benchmark matrix heatmap
# =====================================================================

fig, ax = plt.subplots(figsize=(10.5, 5.2))
data = np.array([[np.nan if v is None else v for v in row] for row in MX], dtype=float)
ax.imshow(data, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(MX_COLS)))
ax.set_xticklabels(MX_COLS, fontsize=11)
ax.set_yticks(range(len(MX_ROWS)))
ax.set_yticklabels(MX_ROWS, fontsize=11)
for i in range(len(MX_ROWS)):
    for j in range(len(MX_COLS)):
        v = MX[i][j]
        if v is None:
            ax.text(j, i, "n/a", ha="center", va="center", fontsize=9, color="#8b949e")
        else:
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9.5,
                    color="white" if v > 0.62 else "#24292f",
                    fontweight="bold" if i == 4 else "normal")
ax.set_title("accuracy by checkpoint and benchmark (R2 row = the collapse; bold row = shipped r2b)",
             fontweight="bold", pad=12, fontsize=12)
ax.set_xlabel("chance floors per column: typed 0.32 / b77 0.01 / emotion 0.17 / AG News 0.25 / WANLI 0.33 / ANLI 0.33",
              fontsize=9, color="#8b949e")
fig.tight_layout()
fig.savefig("assets/matrix.png", dpi=150, transparent=True)
plt.close(fig)

# =====================================================================
# chart 4: Banking77 climb toward Jev (lollipop)
# =====================================================================

fig, ax = plt.subplots(figsize=(10.5, 4.8))
xb = np.arange(len(B77_STEPS))
ax.vlines(xb, 0.80, B77_VALS, color="#c8ccd0", lw=3)
ax.scatter(xb, B77_VALS, s=150, color=ACC[0], zorder=3, edgecolors="#57606a",
           linewidths=0.7)
for i, v in enumerate(B77_VALS):
    ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=9.5,
                fontweight="bold" if i == len(B77_VALS) - 1 else "normal")
ax.axhline(B77_JEV, color=ACC[1], ls="--", lw=1.4)
ax.annotate("Jev (closed API) 0.870  -  72-label config, not controlled",
            (0.02, B77_JEV + 0.002), xycoords=("axes fraction", "data"),
            fontsize=9.5, color="#8b4545")
ax.axhline(0.425, color=ACC[4], ls=":", lw=1.2)
ax.annotate("laya 0.425", (0.02, 0.429), xycoords=("axes fraction", "data"),
            fontsize=9, color="#6e7480")
ax.set_xticks(xb)
ax.set_xticklabels(B77_STEPS, fontsize=9.5)
ax.set_ylim(0.80, 0.885)
ax.set_ylabel("accuracy")
ax.set_title("Banking77: every measured step of the climb (77 labels)",
             fontweight="bold", pad=12)
fig.tight_layout()
fig.savefig("assets/b77climb.png", dpi=150, transparent=True)
plt.close(fig)

# =====================================================================
# chart 5: r2b gamma selection (accuracy flat, ECE U-curve, pick 1.2)
# =====================================================================

fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.plot(GAMMAS, G_ECE, "o-", color=ACC[1], lw=2.2, label="ECE (mix validation)")
ax.plot(GAMMAS, G_ACC, "s--", color=ACC[0], lw=2, label="accuracy (flat)")
ax.scatter([G_PICK], [G_ECE[GAMMAS.index(G_PICK)]], s=260, facecolors="none",
           edgecolors=ACC[3], linewidths=2.2, zorder=4)
ax.annotate(f"selected gamma {G_PICK}\nECE {G_ECE[3]:.4f}", (G_PICK, G_ECE[3]),
            textcoords="offset points", xytext=(14, 16), fontsize=10)
for g, e in zip(GAMMAS, G_ECE):
    ax.annotate(f"{e:.3f}", (g, e), textcoords="offset points", xytext=(0, -16),
                ha="center", fontsize=8.5)
ax.set_xlabel("sharpening exponent gamma")
ax.set_ylabel("metric value")
ax.set_ylim(0, 0.95)
ax.set_title("r2b student: accuracy ignores gamma, calibration picks 1.2",
             fontweight="bold")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("assets/gamma.png", dpi=150, transparent=True)
plt.close(fig)

# =====================================================================
# chart 6: accuracy vs params (published points only)
# =====================================================================

fig, ax = plt.subplots(figsize=(8.5, 5.2))
for name, params, acc, c in PTS:
    ax.scatter(params, acc, s=130, color=c, zorder=3, edgecolors="#57606a",
               linewidths=0.6)
    ax.annotate(name, (params, acc), textcoords="offset points", xytext=OFFS[name],
                fontsize=9)
ax.annotate("Jev: closed API, size not published", (0.98, 0.03),
            xycoords="axes fraction", ha="right", fontsize=9, color="#8b949e",
            style="italic")
ax.set_xlabel("parameters (millions)")
ax.set_ylabel("accuracy")
ax.set_title("accuracy vs model size", fontweight="bold")
ax.set_xlim(0, 4200)
fig.tight_layout()
fig.savefig("assets/params_vs_acc.png", dpi=150, transparent=True)
plt.close(fig)

# =====================================================================
# chart 7: per-workflow accuracy (typed-decisions)
# =====================================================================

fig, ax = plt.subplots(figsize=(9, 4.8))
xw = np.arange(len(WF_LABELS))
ww = 0.34
b1 = ax.bar(xw - ww / 2, WF_LAYA, ww, label="laya (published)", color=ACC[1])
b2 = ax.bar(xw + ww / 2, WF_OEV, ww, label="OEV (4-voter ensemble)", color=ACC[0])
ax.bar_label(b1, fmt="%.3f", fontsize=9, padding=2)
ax.bar_label(b2, fmt="%.3f", fontsize=9, padding=2)
ax.axhline(0.766, color=ACC[1], ls=":", lw=1)
ax.axhline(0.776, color=ACC[0], ls=":", lw=1)
ax.set_xticks(xw)
ax.set_xticklabels(WF_LABELS)
ax.set_ylim(0.6, 0.9)
ax.set_ylabel("accuracy")
ax.set_title("typed-decisions per-workflow: OEV wins 3 of 4", fontweight="bold")
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig("assets/workflows.png", dpi=150, transparent=True)
plt.close(fig)

print("light charts written: benchmarks, zeroshot, transfer_speed, matrix, "
      "b77climb, gamma, params_vs_acc, workflows")

# =====================================================================
# dark variants: same transparent figures, light text and edge swaps
# =====================================================================

DARK_TEXT = "#e6edf3"
DARK_SUB = "#9aa4ae"
DARK_EDGE = "#6e7681"
DARK_LINE = "#3d444d"
plt.rcParams.update({
    "text.color": DARK_TEXT,
    "axes.edgecolor": DARK_EDGE,
    "axes.labelcolor": DARK_TEXT,
    "xtick.color": DARK_TEXT,
    "ytick.color": DARK_TEXT,
    "grid.color": DARK_LINE,
})

def _edge(bars, color):
    for p in bars.patches:
        p.set_edgecolor(color)
        p.set_linewidth(0.4)

# benchmarks (dark)
fig, ax = plt.subplots(figsize=(12, 6.2))
b1 = ax.bar(x - w, JEV, w, label="Jev (closed API, published)", color=ACC[4])
b2 = ax.bar(x, LAYA, w, label="laya (published)", color=ACC[1])
b3 = ax.bar(x + w, OEVD, w, label="OEV (this repo, soup/ensemble)", color=ACC[0])
for b in (b1, b2, b3):
    _edge(b, DARK_TEXT)
ax.set_xticks(x)
ax.set_xticklabels(BENCH)
ax.set_ylim(0, 1.18)
ax.set_ylabel("accuracy")
ax.set_title("OEV vs published numbers on shared public benchmarks (Jev Banking77: 72 labels)",
             fontweight="bold", pad=15)
ax.legend(frameon=False, loc="upper left", fontsize=12)
for b in (b1, b2, b3):
    ax.bar_label(b, fmt="%.3f", fontsize=10, padding=3, color=DARK_TEXT)
fig.tight_layout()
fig.savefig("assets/benchmarks_dark.png", dpi=150, transparent=True)
plt.close(fig)

# zeroshot (dark)
fig, ax = plt.subplots(figsize=(11, 4.6))
for i in range(len(ZS_ROWS)):
    pts = [(vals[i], who) for who, vals, _ in ZS_WHO if vals[i] is not None]
    if pts:
        ax.hlines(i, min(v for v, _ in pts), max(v for v, _ in pts),
                  color=DARK_LINE, lw=2, zorder=1)
    else:
        ax.annotate("all three: not published", (0.995, i),
                    xycoords=("axes fraction", "data"), ha="right", va="center",
                    fontsize=9, color=DARK_SUB, style="italic")
    for v, who in pts:
        ax.scatter(v, i, s=170, color=dict((w, c) for w, _, c in ZS_WHO)[who],
                   zorder=3, edgecolors=DARK_TEXT, linewidths=0.7)
        _zs_annotate(ax, i, v, who, DARK_TEXT)
ax.axvline(0.333, color=DARK_SUB, ls=":", lw=1.2)
ax.annotate("random floor, 3 classes", (0.335, 3.45), fontsize=8.5, color=DARK_SUB)
ax.set_yticks(range(len(ZS_ROWS)))
ax.set_yticklabels(ZS_ROWS, fontsize=11)
ax.set_xlim(0, 1.0)
ax.set_ylim(3.7, -0.7)
ax.invert_yaxis()
ax.set_xlabel("accuracy")
ax.set_title("zero-shot / out-of-domain transfer (dot = measured, gap = not published)",
             fontweight="bold", pad=12)
fig.tight_layout()
fig.savefig("assets/zeroshot_dark.png", dpi=150, transparent=True)
plt.close(fig)

# transfer + latency (dark)
fig, (axt, axl) = plt.subplots(1, 2, figsize=(15.5, 5.2),
                               gridspec_kw={"width_ratios": [1.45, 1]})
for i in range(len(ZS_ROWS)):
    pts = [(vals[i], who) for who, vals, _ in ZS_WHO if vals[i] is not None]
    if pts:
        axt.hlines(i, min(v for v, _ in pts), max(v for v, _ in pts),
                   color=DARK_LINE, lw=2, zorder=1)
    for v, who in pts:
        axt.scatter(v, i, s=140, color=dict((w, c) for w, _, c in ZS_WHO)[who],
                    zorder=3, edgecolors=DARK_TEXT, linewidths=0.7)
        axt.annotate(f"{v:.3f}", (v, i), textcoords="offset points", xytext=(0, 11),
                     ha="center", fontsize=9, color=DARK_TEXT)
axt.axvline(0.333, color=DARK_SUB, ls=":", lw=1.2)
axt.annotate("floor 0.333", (0.34, 3.45), fontsize=8.5, color=DARK_SUB)
axt.set_yticks(range(len(ZS_ROWS)))
axt.set_yticklabels(ZS_ROWS, fontsize=9.5)
axt.set_xlim(0, 1.0)
axt.set_ylim(3.7, -0.7)
axt.invert_yaxis()
axt.set_xlabel("accuracy")
axt.set_title("zero-shot / OOD transfer", fontweight="bold", pad=12, fontsize=13)
axt.legend(handles=_dot_legend(DARK_TEXT), frameon=False, fontsize=9.5,
           loc="lower left", bbox_to_anchor=(0, 1.005), ncols=3,
           columnspacing=1.2, handletextpad=0.2, labelcolor=DARK_TEXT)
lbars = axl.bar(range(len(LAT_LABELS)), LAT_VALS,
                color=[ACC[0], ACC[1], ACC[2], ACC[4]])
_edge(lbars, DARK_TEXT)
axl.set_xticks(range(len(LAT_LABELS)))
axl.set_xticklabels([l.replace("\n", " ") for l in LAT_LABELS], fontsize=9)
axl.bar_label(lbars, fmt="%.1f ms", fontsize=9, padding=3, color=DARK_TEXT)
axl.set_ylim(0, 290)
axl.set_ylabel("ms per question")
axl.set_title("latency (lower is better; ranges use midpoint)", fontweight="bold",
              pad=12, fontsize=13)
fig.tight_layout()
fig.savefig("assets/transfer_speed_dark.png", dpi=150, transparent=True)
plt.close(fig)

# matrix (dark)
fig, ax = plt.subplots(figsize=(10.5, 5.2))
ax.imshow(data, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(MX_COLS)))
ax.set_xticklabels(MX_COLS, fontsize=11)
ax.set_yticks(range(len(MX_ROWS)))
ax.set_yticklabels(MX_ROWS, fontsize=11)
for i in range(len(MX_ROWS)):
    for j in range(len(MX_COLS)):
        v = MX[i][j]
        if v is None:
            ax.text(j, i, "n/a", ha="center", va="center", fontsize=9, color=DARK_SUB)
        else:
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9.5,
                    color="white" if v > 0.62 else DARK_TEXT,
                    fontweight="bold" if i == 4 else "normal")
for j, fl in enumerate(MX_FLOOR):
    pass
ax.set_title("accuracy by checkpoint and benchmark (R2 row = the collapse; bold row = shipped r2b)",
             fontweight="bold", pad=12, fontsize=12)
ax.set_xlabel("chance floors per column: typed 0.32 / b77 0.01 / emotion 0.17 / AG News 0.25 / WANLI 0.33 / ANLI 0.33",
              fontsize=9, color="#8b949e")
fig.tight_layout()
fig.savefig("assets/matrix_dark.png", dpi=150, transparent=True)
plt.close(fig)

# b77 climb (dark)
fig, ax = plt.subplots(figsize=(10.5, 4.8))
ax.vlines(xb, 0.80, B77_VALS, color=DARK_LINE, lw=3)
ax.scatter(xb, B77_VALS, s=150, color=ACC[0], zorder=3, edgecolors=DARK_TEXT,
           linewidths=0.7)
for i, v in enumerate(B77_VALS):
    ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=9.5, color=DARK_TEXT,
                fontweight="bold" if i == len(B77_VALS) - 1 else "normal")
ax.axhline(B77_JEV, color=ACC[1], ls="--", lw=1.4)
ax.annotate("Jev (closed API) 0.870  -  72-label config, not controlled",
            (0.02, B77_JEV + 0.002), xycoords=("axes fraction", "data"),
            fontsize=9.5, color="#e0a0a0")
ax.axhline(0.425, color=ACC[4], ls=":", lw=1.2)
ax.annotate("laya 0.425", (0.02, 0.429), xycoords=("axes fraction", "data"),
            fontsize=9, color=DARK_SUB)
ax.set_xticks(xb)
ax.set_xticklabels(B77_STEPS, fontsize=9.5)
ax.set_ylim(0.80, 0.885)
ax.set_ylabel("accuracy")
ax.set_title("Banking77: every measured step of the climb (77 labels)",
             fontweight="bold", pad=12)
fig.tight_layout()
fig.savefig("assets/b77climb_dark.png", dpi=150, transparent=True)
plt.close(fig)

# gamma (dark)
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.plot(GAMMAS, G_ECE, "o-", color=ACC[1], lw=2.2, label="ECE (mix validation)")
ax.plot(GAMMAS, G_ACC, "s--", color=ACC[0], lw=2, label="accuracy (flat)")
ax.scatter([G_PICK], [G_ECE[GAMMAS.index(G_PICK)]], s=260, facecolors="none",
           edgecolors=ACC[3], linewidths=2.2, zorder=4)
ax.annotate(f"selected gamma {G_PICK}\nECE {G_ECE[3]:.4f}", (G_PICK, G_ECE[3]),
            textcoords="offset points", xytext=(14, 16), fontsize=10, color=DARK_TEXT)
for g, e in zip(GAMMAS, G_ECE):
    ax.annotate(f"{e:.3f}", (g, e), textcoords="offset points", xytext=(0, -16),
                ha="center", fontsize=8.5, color=DARK_TEXT)
ax.set_xlabel("sharpening exponent gamma")
ax.set_ylabel("metric value")
ax.set_ylim(0, 0.95)
ax.set_title("r2b student: accuracy ignores gamma, calibration picks 1.2",
             fontweight="bold")
ax.legend(frameon=False, loc="upper right", labelcolor=DARK_TEXT)
fig.tight_layout()
fig.savefig("assets/gamma_dark.png", dpi=150, transparent=True)
plt.close(fig)

# params vs acc (dark)
fig, ax = plt.subplots(figsize=(8.5, 5.2))
for name, params, acc, c in PTS:
    ax.scatter(params, acc, s=130, color=c, zorder=3, edgecolors=DARK_TEXT,
               linewidths=0.6)
    ax.annotate(name, (params, acc), textcoords="offset points", xytext=OFFS[name],
                fontsize=9, color=DARK_TEXT)
ax.annotate("Jev: closed API, size not published", (0.98, 0.03),
            xycoords="axes fraction", ha="right", fontsize=9, color=DARK_SUB,
            style="italic")
ax.set_xlabel("parameters (millions)")
ax.set_ylabel("accuracy")
ax.set_title("accuracy vs model size", fontweight="bold")
ax.set_xlim(0, 4200)
fig.tight_layout()
fig.savefig("assets/params_vs_acc_dark.png", dpi=150, transparent=True)
plt.close(fig)

# workflows (dark)
fig, ax = plt.subplots(figsize=(9, 4.8))
b1 = ax.bar(xw - ww / 2, WF_LAYA, ww, label="laya (published)", color=ACC[1])
b2 = ax.bar(xw + ww / 2, WF_OEV, ww, label="OEV (4-voter ensemble)", color=ACC[0])
for b in (b1, b2):
    _edge(b, DARK_TEXT)
ax.bar_label(b1, fmt="%.3f", fontsize=9, padding=2, color=DARK_TEXT)
ax.bar_label(b2, fmt="%.3f", fontsize=9, padding=2, color=DARK_TEXT)
ax.axhline(0.766, color=ACC[1], ls=":", lw=1)
ax.axhline(0.776, color=ACC[0], ls=":", lw=1)
ax.set_xticks(xw)
ax.set_xticklabels(WF_LABELS)
ax.set_ylim(0.6, 0.9)
ax.set_ylabel("accuracy")
ax.set_title("typed-decisions per-workflow: OEV wins 3 of 4", fontweight="bold")
ax.legend(frameon=False, loc="lower left", labelcolor=DARK_TEXT)
fig.tight_layout()
fig.savefig("assets/workflows_dark.png", dpi=150, transparent=True)
plt.close(fig)

print("dark variants written (transparent)")
