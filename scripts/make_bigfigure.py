"""Laya-style multi-panel benchmark figure from measured OEV numbers.

Palette and typography come from chartstyle.py; every measured/published
number comes from chartdata.py, which cross-checks registered OEV values
against docs/claims.json at import time. Transparent backgrounds (page shows
through); pastel palette; dark text for the light variant, light text for the
dark variant.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chartstyle as cs
import chartdata

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# fail loudly at import if chart numbers and the claims registry diverged
chartdata.load()

os.makedirs("assets", exist_ok=True)

BLUE, RED, GREEN, GOLD, GRAY = cs.P_BLUE, cs.P_RED, cs.P_GREEN, "#e8d5a3", cs.P_GRAY
TEXT, EDGE, SUB, LINE = cs.TEXT, cs.EDGE, cs.SUB, cs.LINE
DTEXT, DEDGE, DSUB, DLINE = cs.DARK_TEXT, cs.DARK_EDGE, cs.DARK_SUB, cs.DARK_LINE
GRAY_ANNOT = cs.GRAY_ANNOT

plt.rcParams.update(cs.BASE_RC)
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
})

# =====================================================================
# data: single source in chartdata (competitor rows quoted from published
# tables; OEV rows verified against docs/claims.json by chartdata.load())
# =====================================================================

bench = chartdata.BENCH
jev = chartdata.JEV
laya = chartdata.LAYA
oev = chartdata.OEVD
names = chartdata.BASELINE_NAMES
vals = chartdata.BASELINE_VALS
wf = chartdata.WF_LABELS
laya_wf = chartdata.WF_LAYA
oev_wf = chartdata.WF_OEV
qs = chartdata.OEV_LAT_BATCH_LABELS
oev_lat = chartdata.OEV_LAT
models = ["laya-typed\ndecisions", "OEV single", "OEV ensemble\nsharpened", "Jev"]
ece = chartdata.ECE_PANEL
gam = ["g=1.0", "g=1.5", "g=2.0", "g=2.5"]
soft = chartdata.GAMMA_SOFT
hard = chartdata.GAMMA_HARD

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)

# ---------- Panel 1: accuracy on public datasets ----------
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(bench))
w = 0.26
b1 = ax.bar(x - w, jev, w, label="Jev (published)", color=GRAY)
b2 = ax.bar(x, laya, w, label="laya (published)", color=RED)
b3 = ax.bar(x + w, oev, w, label="OEV (this repo)", color=BLUE)
for b in (b1, b2, b3):
    cs.edge_bars(b, EDGE)
    ax.bar_label(b, fmt="%.3f", fontsize=6)
ax.set_xticks(x)
ax.set_xticklabels(bench, fontsize=7)
ax.set_ylim(0, 1.15)
ax.set_ylabel("accuracy")
ax.set_title("Accuracy - shared public datasets")
ax.legend(frameon=False, fontsize=7, loc="lower right")

# ---------- Panel 2: typed-decisions vs baselines ----------
ax = fig.add_subplot(gs[0, 1])
cols = [GRAY, GRAY, GOLD, RED, RED, BLUE]
bars = ax.bar(names, vals, color=cols)
cs.edge_bars(bars, EDGE)
ax.bar_label(bars, fmt="%.4f", fontsize=7)
ax.axhline(0.735, color=GOLD, ls=":", lw=1)
ax.set_ylim(0, 0.9)
ax.set_ylabel("accuracy")
ax.set_title("typed-decisions: vs baselines and ceiling")
ax.tick_params(axis="x", labelsize=7)

# ---------- Panel 3: per-workflow (horizontal bars) ----------
ax = fig.add_subplot(gs[0, 2])
y = np.arange(len(wf))
h = 0.35
b1 = ax.barh(y - h / 2, laya_wf, h, label="laya", color=RED)
b2 = ax.barh(y + h / 2, oev_wf, h, label="OEV", color=BLUE)
for b in (b1, b2):
    cs.edge_bars(b, EDGE)
    ax.bar_label(b, fmt="%.3f", fontsize=6)
ax.set_yticks(y)
ax.set_yticklabels(wf, fontsize=7)
ax.set_xlim(0.5, 1.0)
ax.set_xlabel("accuracy")
ax.set_title("typed-decisions: every workflow")
ax.legend(frameon=False, fontsize=7, loc="lower right")

# ---------- Panel 4: speed on one T4 (measured points only) ----------
ax = fig.add_subplot(gs[1, 0])
# measured: OEV p50 22.2ms at 1 question, 15.9ms/q at batch 32.
# laya: published 39.5ms single (English). No measured multi-question curve.
b2 = ax.bar(qs, oev_lat, 0.45, label="OEV (measured)", color=BLUE)
cs.edge_bars(b2, EDGE)
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
bars = ax.bar(models, ece, color=[RED, BLUE, BLUE, GRAY])
cs.edge_bars(bars, EDGE)
ax.bar_label(bars, fmt="%.4f", fontsize=7)
ax.set_ylabel("mean ECE (lower is better)")
ax.set_ylim(0, 0.3)
ax.set_title("Calibration (post-temperature)")

# ---------- Panel 6: accuracy vs size ----------
ax = fig.add_subplot(gs[1, 2])
pts = [(name, params, acc, chartdata.oev_color(color)) for name, params, acc, color in chartdata.BIGFIG_PTS]
POFF = chartdata.BIGFIG_POFF
for name, params, acc, c in pts:
    ax.scatter(params, acc, s=120, color=c, zorder=3, edgecolors=EDGE, linewidths=0.5)
    ax.annotate(name, (params, acc), textcoords="offset points", xytext=POFF[name], fontsize=7)
ax.annotate("Jev: closed API, size not published", (0.97, 0.02),
            xycoords="axes fraction", ha="right", fontsize=7, color=SUB, style="italic")
ax.set_xlim(0, 4200)
ax.set_ylim(0.72, 0.90)
ax.set_xlabel("parameters (millions)")
ax.set_ylabel("typed-decisions accuracy")
ax.set_title("Accuracy vs model size")

# ---------- Panel 7: soft accuracy sharpening (typed ensemble) ----------
ax = fig.add_subplot(gs[2, :])
b1 = ax.bar(np.arange(4) - 0.2, hard, 0.4, label="hard accuracy", color=GREEN)
b2 = ax.bar(np.arange(4) + 0.2, soft, 0.4, label="soft accuracy", color=BLUE)
for b in (b1, b2):
    cs.edge_bars(b, EDGE)
    ax.bar_label(b, fmt="%.4f", fontsize=8)
ax.axhline(0.471, color=RED, ls=":", lw=1.5)
ax.annotate("laya soft acc 0.471", (2.6, 0.48), color="#8b4545", fontsize=8)
ax.axhline(0.580, color=GRAY, ls=":", lw=1.5)
ax.annotate("Jev soft acc 0.580", (0.0, 0.59), color=SUB, fontsize=8)
ax.set_xticks(np.arange(4))
ax.set_xticklabels(gam)
ax.set_ylim(0.3, 0.85)
ax.set_title("Soft-accuracy sharpening sweep (typed ensemble, exploratory)")
ax.legend(frameon=False, fontsize=8, loc="lower right")

fig.suptitle("OEV vs TypeSafe Jev and laya: accuracy, workflows, speed, calibration, size - all measured",
             fontsize=13, fontweight="bold")
fig.savefig("assets/oev_vs_jev_full.png", dpi=150, bbox_inches="tight", transparent=True)
plt.close(fig)

print("written: assets/oev_vs_jev_full.png")


# ---------------- dark variant ----------------
cs.apply_dark()

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)

ax = fig.add_subplot(gs[0, 0])
b1 = ax.bar(x - w, jev, w, label="Jev (published)", color=GRAY)
b2 = ax.bar(x, laya, w, label="laya (published)", color=RED)
b3 = ax.bar(x + w, oev, w, label="OEV (this repo)", color=BLUE)
for b in (b1, b2, b3):
    cs.edge_bars(b, DTEXT)
    ax.bar_label(b, fmt="%.3f", fontsize=6, color=DTEXT)
ax.set_xticks(x)
ax.set_xticklabels(bench, fontsize=7)
ax.set_ylim(0, 1.15)
ax.set_ylabel("accuracy")
ax.set_title("Accuracy - shared public datasets")
ax.legend(frameon=False, fontsize=7, loc="lower right")

ax = fig.add_subplot(gs[0, 1])
bars = ax.bar(names, vals, color=cols)
cs.edge_bars(bars, DTEXT)
ax.bar_label(bars, fmt="%.4f", fontsize=7, color=DTEXT)
ax.axhline(0.735, color=GOLD, ls=":", lw=1)
ax.set_ylim(0, 0.9)
ax.set_ylabel("accuracy")
ax.set_title("typed-decisions: vs baselines and ceiling")
ax.tick_params(axis="x", labelsize=7)

ax = fig.add_subplot(gs[0, 2])
b1 = ax.barh(y - h / 2, laya_wf, h, label="laya", color=RED)
b2 = ax.barh(y + h / 2, oev_wf, h, label="OEV", color=BLUE)
for b in (b1, b2):
    cs.edge_bars(b, DTEXT)
    ax.bar_label(b, fmt="%.3f", fontsize=6, color=DTEXT)
ax.set_yticks(y)
ax.set_yticklabels(wf, fontsize=7)
ax.set_xlim(0.5, 1.0)
ax.set_xlabel("accuracy")
ax.set_title("typed-decisions: every workflow")
ax.legend(frameon=False, fontsize=7, loc="lower right")

ax = fig.add_subplot(gs[1, 0])
b2 = ax.bar(qs, oev_lat, 0.45, label="OEV (measured)", color=BLUE)
cs.edge_bars(b2, DTEXT)
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
cs.edge_bars(bars, DTEXT)
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
ax.set_xlim(0, 4200)
ax.set_ylim(0.72, 0.90)
ax.set_xlabel("parameters (millions)")
ax.set_ylabel("typed-decisions accuracy")
ax.set_title("Accuracy vs model size")

ax = fig.add_subplot(gs[2, :])
b1 = ax.bar(np.arange(4) - 0.2, hard, 0.4, label="hard accuracy", color=GREEN)
b2 = ax.bar(np.arange(4) + 0.2, soft, 0.4, label="soft accuracy", color=BLUE)
for b in (b1, b2):
    cs.edge_bars(b, DTEXT)
    ax.bar_label(b, fmt="%.4f", fontsize=8, color=DTEXT)
ax.axhline(0.471, color=RED, ls=":", lw=1.5)
ax.annotate("laya soft acc 0.471", (2.6, 0.48), color="#e0a0a0", fontsize=8)
ax.axhline(0.580, color=GRAY, ls=":", lw=1.5)
ax.annotate("Jev soft acc 0.580", (0.0, 0.59), color=DSUB, fontsize=8)
ax.set_xticks(np.arange(4))
ax.set_xticklabels(gam)
ax.set_ylim(0.3, 0.85)
ax.set_title("Soft-accuracy sharpening sweep (typed ensemble, exploratory)")
ax.legend(frameon=False, fontsize=8, loc="lower right")

fig.suptitle("OEV vs TypeSafe Jev and laya: accuracy, workflows, speed, calibration, size - all measured",
             fontsize=13, fontweight="bold")
fig.savefig("assets/oev_vs_jev_full_dark.png", dpi=150, bbox_inches="tight", transparent=True)
plt.close(fig)

print("written: assets/oev_vs_jev_full_dark.png")
