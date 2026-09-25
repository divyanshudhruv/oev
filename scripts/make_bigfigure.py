"""Laya-style multi-panel benchmark figure from measured OEV numbers.

Transparent backgrounds (page shows through); pastel palette; dark text
for the light variant, light text for the dark variant.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("assets", exist_ok=True)

BLUE, RED, GREEN, GOLD, GRAY = "#a8c5e6", "#f0a8a8", "#b5d4bf", "#e8d5a3", "#b0b8c0"
TEXT, EDGE, SUB, LINE = "#24292f", "#57606a", "#8b949e", "#c8ccd0"

plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.transparent": True,
    "axes.grid": True,
    "grid.color": LINE,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.6,
    "text.color": TEXT,
    "axes.edgecolor": EDGE,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)

# ---------- Panel 1: accuracy on public datasets ----------
ax = fig.add_subplot(gs[0, 0])
bench = ["typed-decisions", "AG News", "emotion (zero-shot)", "Banking77"]
jev = [0.727, 0.910, 0.480, 0.870]
laya = [0.766, 0.950, 0.595, 0.425]
oev = [0.7760, 0.9489, 0.6505, 0.8584]  # emotion = zero-shot student; b77 = soup (single 735MB file)
x = np.arange(len(bench)); w = 0.26
b1 = ax.bar(x - w, jev, w, label="Jev (published)", color=GRAY)
b2 = ax.bar(x, laya, w, label="laya (published)", color=RED)
b3 = ax.bar(x + w, oev, w, label="OEV (this repo)", color=BLUE)
for b in (b1, b2, b3):
    for p in b.patches:
        p.set_edgecolor(EDGE)
        p.set_linewidth(0.4)
    ax.bar_label(b, fmt="%.3f", fontsize=6)
ax.set_xticks(x); ax.set_xticklabels(bench, fontsize=7)
ax.set_ylim(0, 1.15); ax.set_ylabel("accuracy")
ax.set_title("Accuracy - shared public datasets")
ax.legend(frameon=False, fontsize=7, loc="lower right")

# ---------- Panel 2: typed-decisions vs baselines ----------
ax = fig.add_subplot(gs[0, 1])
names = ["random\nguess", "majority\nclass", "teacher\nceiling", "Jev", "laya-typed\ndecisions", "OEV\nensemble"]
vals = [0.318, 0.461, 0.735, 0.727, 0.766, 0.7760]
cols = [GRAY, GRAY, GOLD, RED, RED, BLUE]
bars = ax.bar(names, vals, color=cols)
for p in bars.patches:
    p.set_edgecolor(EDGE)
    p.set_linewidth(0.4)
ax.bar_label(bars, fmt="%.4f", fontsize=7)
ax.axhline(0.735, color=GOLD, ls=":", lw=1)
ax.set_ylim(0, 0.9); ax.set_ylabel("accuracy")
ax.set_title("typed-decisions: vs baselines and ceiling")
ax.tick_params(axis="x", labelsize=7)

# ---------- Panel 3: per-workflow (horizontal bars) ----------
ax = fig.add_subplot(gs[0, 2])
wf = ["security\nincidents", "agent-trace\nobservability", "customer\nservice", "invoice\nprocessing"]
laya_wf = [0.766, 0.730, 0.764, 0.804]
oev_wf = [0.7220, 0.7400, 0.8040, 0.8360]
y = np.arange(len(wf)); h = 0.35
b1 = ax.barh(y - h/2, laya_wf, h, label="laya", color=RED)
b2 = ax.barh(y + h/2, oev_wf, h, label="OEV", color=BLUE)
for b in (b1, b2):
    for p in b.patches:
        p.set_edgecolor(EDGE)
        p.set_linewidth(0.4)
    ax.bar_label(b, fmt="%.3f", fontsize=6)
ax.set_yticks(y); ax.set_yticklabels(wf, fontsize=7)
ax.set_xlim(0.5, 1.0); ax.set_xlabel("accuracy")
ax.set_title("typed-decisions: every workflow")
ax.legend(frameon=False, fontsize=7, loc="lower right")

# ---------- Panel 4: speed on one T4 (measured points only) ----------
ax = fig.add_subplot(gs[1, 0])
# measured: OEV p50 22.2ms at 1 question, 15.9ms/q at batch 32.
# laya: published 39.5ms single (English). No measured multi-question curve.
qs = ["1 question", "batch 32\n(per question)"]
oev_lat = [22.2, 15.9]
b2 = ax.bar(qs, oev_lat, 0.45, label="OEV (measured)", color=BLUE)
for p in b2.patches:
    p.set_edgecolor(EDGE)
    p.set_linewidth(0.4)
ax.bar_label(b2, fmt="%.1f ms", fontsize=8)
ax.axhline(39.5, color=RED, ls=":", lw=1.5)
ax.annotate("laya single-question p50 39.5 (published)", (0.02, 40.6),
            xycoords=("axes fraction", "data"), fontsize=7.5, color="#8b4545")
ax.set_ylabel("p50 latency (ms)")
ax.set_ylim(0, 50)
ax.set_title("Speed on one T4 (measured points)")
ax.legend(frameon=False, fontsize=7)

# ---------- Panel 5: calibration ----------
ax = fig.add_subplot(gs[1, 1])
models = ["laya-typed\ndecisions", "OEV single", "OEV ensemble\nsharpened", "Jev"]
ece = [0.213, 0.0938, 0.0298, 0.144]
bars = ax.bar(models, ece, color=[RED, BLUE, BLUE, GRAY])
for p in bars.patches:
    p.set_edgecolor(EDGE)
    p.set_linewidth(0.4)
ax.bar_label(bars, fmt="%.4f", fontsize=7)
ax.set_ylabel("mean ECE (lower is better)")
ax.set_ylim(0, 0.3)
ax.set_title("Calibration (post-temperature)")

# ---------- Panel 6: accuracy vs size ----------
ax = fig.add_subplot(gs[1, 2])
pts = [
    ("laya (421M)", 421, 0.766, RED),
    ("OEV single (184M)", 184, 0.7705, BLUE),
    ("OEV ensemble (736M)", 736, 0.7760, BLUE),
    ("OEV b77 soup (184M)", 184, 0.8584, BLUE),
    ("Kev-0.8B (new sources)", 800, 0.697, GREEN),
    ("Kev-4B (new sources)", 4000, 0.838, GREEN),
    ("Kev-9B (new sources)", 9000, 0.852, GREEN),
    ("Kev-27B (new sources)", 27000, 0.896, GREEN),
]
POFF = {"laya (421M)": (6, -12), "OEV single (184M)": (6, -12),
        "OEV ensemble (736M)": (6, 6), "OEV b77 soup (184M)": (-20, 8),
        "Kev-0.8B (new sources)": (0, 8), "Kev-4B (new sources)": (-40, 8),
        "Kev-9B (new sources)": (-10, 8), "Kev-27B (new sources)": (-60, 8)}
for name, params, acc, c in pts:
    ax.scatter(params, acc, s=120, color=c, zorder=3, edgecolors=EDGE, linewidths=0.5)
    ax.annotate(name, (params, acc), textcoords="offset points", xytext=POFF[name], fontsize=7)
ax.annotate("Jev: closed API, size not published", (0.97, 0.02),
            xycoords="axes fraction", ha="right", fontsize=7, color=SUB, style="italic")
ax.set_xlim(0, 4200); ax.set_ylim(0.72, 0.90)
ax.set_xlabel("parameters (millions)"); ax.set_ylabel("typed-decisions accuracy")
ax.set_title("Accuracy vs model size")

# ---------- Panel 7: soft accuracy sharpening (typed ensemble) ----------
ax = fig.add_subplot(gs[2, :])
gam = ["g=1.0", "g=1.5", "g=2.0", "g=2.5"]
soft = [0.5871, 0.6410, 0.6758, 0.7020]
hard = [0.7755, 0.7750, 0.7750, 0.7730]
b1 = ax.bar(np.arange(4) - 0.2, hard, 0.4, label="hard accuracy", color=GREEN)
b2 = ax.bar(np.arange(4) + 0.2, soft, 0.4, label="soft accuracy", color=BLUE)
for b in (b1, b2):
    for p in b.patches:
        p.set_edgecolor(EDGE)
        p.set_linewidth(0.4)
    ax.bar_label(b, fmt="%.4f", fontsize=8)
ax.axhline(0.471, color=RED, ls=":", lw=1.5)
ax.annotate("laya soft acc 0.471", (2.6, 0.48), color="#8b4545", fontsize=8)
ax.axhline(0.580, color=GRAY, ls=":", lw=1.5)
ax.annotate("Jev soft acc 0.580", (0.0, 0.59), color=SUB, fontsize=8)
ax.set_xticks(np.arange(4)); ax.set_xticklabels(gam)
ax.set_ylim(0.3, 0.85)
ax.set_title("Soft-accuracy sharpening sweep (typed ensemble, exploratory)")
ax.legend(frameon=False, fontsize=8, loc="lower right")

fig.suptitle("OEV vs TypeSafe Jev and laya: accuracy, workflows, speed, calibration, size - all measured",
             fontsize=13, fontweight="bold")
fig.savefig("assets/oev_vs_jev_full.png", dpi=150, bbox_inches="tight", transparent=True)
plt.close(fig)

print("written: assets/oev_vs_jev_full.png")


# ---------------- dark variant ----------------
DTEXT, DEDGE, DSUB, DLINE = "#e6edf3", "#6e7681", "#9aa4ae", "#3d444d"

plt.rcParams.update({
    "text.color": DTEXT,
    "axes.edgecolor": DEDGE,
    "axes.labelcolor": DTEXT,
    "xtick.color": DTEXT,
    "ytick.color": DTEXT,
    "grid.color": DLINE,
})

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)

ax = fig.add_subplot(gs[0, 0])
b1 = ax.bar(x - w, jev, w, label="Jev (published)", color=GRAY)
b2 = ax.bar(x, laya, w, label="laya (published)", color=RED)
b3 = ax.bar(x + w, oev, w, label="OEV (this repo)", color=BLUE)
for b in (b1, b2, b3):
    for p in b.patches:
        p.set_edgecolor(DTEXT)
        p.set_linewidth(0.4)
    ax.bar_label(b, fmt="%.3f", fontsize=6, color=DTEXT)
ax.set_xticks(x); ax.set_xticklabels(bench, fontsize=7)
ax.set_ylim(0, 1.15); ax.set_ylabel("accuracy")
ax.set_title("Accuracy - shared public datasets")
ax.legend(frameon=False, fontsize=7, loc="lower right")

ax = fig.add_subplot(gs[0, 1])
bars = ax.bar(names, vals, color=cols)
for p in bars.patches:
    p.set_edgecolor(DTEXT)
    p.set_linewidth(0.4)
ax.bar_label(bars, fmt="%.4f", fontsize=7, color=DTEXT)
ax.axhline(0.735, color=GOLD, ls=":", lw=1)
ax.set_ylim(0, 0.9); ax.set_ylabel("accuracy")
ax.set_title("typed-decisions: vs baselines and ceiling")
ax.tick_params(axis="x", labelsize=7)

ax = fig.add_subplot(gs[0, 2])
b1 = ax.barh(y - h/2, laya_wf, h, label="laya", color=RED)
b2 = ax.barh(y + h/2, oev_wf, h, label="OEV", color=BLUE)
for b in (b1, b2):
    for p in b.patches:
        p.set_edgecolor(DTEXT)
        p.set_linewidth(0.4)
    ax.bar_label(b, fmt="%.3f", fontsize=6, color=DTEXT)
ax.set_yticks(y); ax.set_yticklabels(wf, fontsize=7)
ax.set_xlim(0.5, 1.0); ax.set_xlabel("accuracy")
ax.set_title("typed-decisions: every workflow")
ax.legend(frameon=False, fontsize=7, loc="lower right")

ax = fig.add_subplot(gs[1, 0])
b2 = ax.bar(qs, oev_lat, 0.45, label="OEV (measured)", color=BLUE)
for p in b2.patches:
    p.set_edgecolor(DTEXT)
    p.set_linewidth(0.4)
ax.bar_label(b2, fmt="%.1f ms", fontsize=8, color=DTEXT)
ax.axhline(39.5, color=RED, ls=":", lw=1.5)
ax.annotate("laya single-question p50 39.5 (published)", (0.02, 40.6),
            xycoords=("axes fraction", "data"), fontsize=7.5, color="#e0a0a0")
ax.set_ylabel("p50 latency (ms)")
ax.set_ylim(0, 50)
ax.set_title("Speed on one T4 (measured points)")
ax.legend(frameon=False, fontsize=7)

ax = fig.add_subplot(gs[1, 1])
bars = ax.bar(models, ece, color=[RED, BLUE, BLUE, GRAY])
for p in bars.patches:
    p.set_edgecolor(DTEXT)
    p.set_linewidth(0.4)
ax.bar_label(bars, fmt="%.4f", fontsize=7, color=DTEXT)
ax.set_ylabel("mean ECE (lower is better)")
ax.set_ylim(0, 0.3)
ax.set_title("Calibration (post-temperature)")

ax = fig.add_subplot(gs[1, 2])
for name, params, acc, c in pts:
    ax.scatter(params, acc, s=120, color=c, zorder=3, edgecolors=DTEXT, linewidths=0.5)
    ax.annotate(name, (params, acc), textcoords="offset points", xytext=POFF[name],
                fontsize=7, color=DTEXT)
ax.annotate("Jev: closed API, size not published", (0.97, 0.02),
            xycoords="axes fraction", ha="right", fontsize=7, color=DSUB, style="italic")
ax.set_xlim(0, 4200); ax.set_ylim(0.72, 0.90)
ax.set_xlabel("parameters (millions)"); ax.set_ylabel("typed-decisions accuracy")
ax.set_title("Accuracy vs model size")

ax = fig.add_subplot(gs[2, :])
b1 = ax.bar(np.arange(4) - 0.2, hard, 0.4, label="hard accuracy", color=GREEN)
b2 = ax.bar(np.arange(4) + 0.2, soft, 0.4, label="soft accuracy", color=BLUE)
for b in (b1, b2):
    for p in b.patches:
        p.set_edgecolor(DTEXT)
        p.set_linewidth(0.4)
    ax.bar_label(b, fmt="%.4f", fontsize=8, color=DTEXT)
ax.axhline(0.471, color=RED, ls=":", lw=1.5)
ax.annotate("laya soft acc 0.471", (2.6, 0.48), color="#e0a0a0", fontsize=8)
ax.axhline(0.580, color=GRAY, ls=":", lw=1.5)
ax.annotate("Jev soft acc 0.580", (0.0, 0.59), color=DSUB, fontsize=8)
ax.set_xticks(np.arange(4)); ax.set_xticklabels(gam)
ax.set_ylim(0.3, 0.85)
ax.set_title("Soft-accuracy sharpening sweep (typed ensemble, exploratory)")
ax.legend(frameon=False, fontsize=8, loc="lower right")

fig.suptitle("OEV vs TypeSafe Jev and laya: accuracy, workflows, speed, calibration, size - all measured",
             fontsize=13, fontweight="bold")
fig.savefig("assets/oev_vs_jev_full_dark.png", dpi=150, bbox_inches="tight", transparent=True)
plt.close(fig)

print("written: assets/oev_vs_jev_full_dark.png")
