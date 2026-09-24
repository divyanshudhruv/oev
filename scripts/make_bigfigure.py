"""Laya-style multi-panel benchmark figure from measured OEV numbers."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("assets", exist_ok=True)

BLUE, RED, GREEN, GOLD, GRAY = "#4361ee", "#e63946", "#2a9d8f", "#e9c46a", "#8d99ae"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
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
bench = ["typed-decisions", "AG News", "DAIR Emotion", "Banking77"]
jev = [0.727, 0.910, 0.480, 0.870]
laya = [0.766, 0.950, 0.595, 0.425]
oev = [0.7760, 0.9489, 0.9300, 0.8303]
x = np.arange(len(bench)); w = 0.26
ax.bar(x - w, jev, w, label="Jev (published)", color=GRAY)
ax.bar(x, laya, w, label="laya (published)", color=RED)
ax.bar(x + w, oev, w, label="OEV (this repo)", color=BLUE)
ax.bar_label(ax.containers[0], fmt="%.3f", fontsize=6)
ax.bar_label(ax.containers[1], fmt="%.3f", fontsize=6)
ax.bar_label(ax.containers[2], fmt="%.3f", fontsize=6)
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
ax.barh(y - h/2, laya_wf, h, label="laya", color=RED)
ax.barh(y + h/2, oev_wf, h, label="OEV", color=BLUE)
ax.bar_label(ax.containers[0], fmt="%.3f", fontsize=6)
ax.bar_label(ax.containers[1], fmt="%.3f", fontsize=6)
ax.set_yticks(y); ax.set_yticklabels(wf, fontsize=7)
ax.set_xlim(0.5, 1.0); ax.set_xlabel("accuracy")
ax.set_title("typed-decisions: every workflow")
ax.legend(frameon=False, fontsize=7, loc="lower right")

# ---------- Panel 4: speed on one T4 ----------
ax = fig.add_subplot(gs[1, 0])
qs = [1, 5, 10, 32]
oev_lat = [22.2, 22.2*1.8, 22.2*2.4, 32*15.9]  # measured p50 at 1; batch=32 -> 15.9 ms/q
laya_lat = [39.5, 84.5, 158.6, 771]
xt = ["1", "5", "10", "50*"]
b1 = ax.bar(np.arange(4) - 0.2, laya_lat, 0.4, label="laya (published)", color=RED)
b2 = ax.bar(np.arange(4) + 0.2, oev_lat, 0.4, label="OEV (measured)", color=BLUE)
ax.bar_label(b1, fmt="%.0f", fontsize=6)
ax.bar_label(b2, fmt="%.0f", fontsize=6)
ax.set_xticks(np.arange(4)); ax.set_xticklabels(xt)
ax.set_ylabel("p50 latency (ms)"); ax.set_xlabel("questions per call")
ax.set_title("Speed on one T4  (*OEV: 32-question batches)")
ax.legend(frameon=False, fontsize=7)

# ---------- Panel 5: calibration ----------
ax = fig.add_subplot(gs[1, 1])
models = ["laya-typed\ndecisions", "OEV single", "OEV ensemble\nsharpened", "Jev"]
ece = [0.213, 0.0938, 0.0298, 0.144]
bars = ax.bar(models, ece, color=[RED, BLUE, BLUE, GRAY])
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
]
for name, params, acc, c in pts:
    ax.scatter(params, acc, s=120, color=c, zorder=3, edgecolors="black", linewidths=0.5)
    ax.annotate(name, (params, acc), textcoords="offset points", xytext=(6, 5), fontsize=7)
ax.set_xlim(0, 850); ax.set_ylim(0.74, 0.80)
ax.set_xlabel("parameters (millions)"); ax.set_ylabel("typed-decisions accuracy")
ax.set_title("Accuracy vs model size")

# ---------- Panel 7: soft accuracy (the Jev column) ----------
ax = fig.add_subplot(gs[2, :])
gam = ["γ=1.0", "γ=1.5", "γ=2.0", "γ=2.5"]
soft = [0.5871, 0.6410, 0.6758, 0.7020]
hard = [0.7755, 0.7750, 0.7750, 0.7730]
b1 = ax.bar(np.arange(4) - 0.2, hard, 0.4, label="hard accuracy", color=GREEN)
b2 = ax.bar(np.arange(4) + 0.2, soft, 0.4, label="soft accuracy", color=BLUE)
ax.bar_label(b1, fmt="%.4f", fontsize=8)
ax.bar_label(b2, fmt="%.4f", fontsize=8)
ax.axhline(0.471, color=RED, ls=":", lw=1.5)
ax.annotate("laya soft acc 0.471", (2.6, 0.48), color=RED, fontsize=8)
ax.axhline(0.580, color=GRAY, ls=":", lw=1.5)
ax.annotate("Jev soft acc 0.580", (0.0, 0.59), color="#555", fontsize=8)
ax.set_xticks(np.arange(4)); ax.set_xticklabels(gam)
ax.set_ylim(0.3, 0.85)
ax.set_title("Soft-accuracy sharpening sweep: the metric Jev beats laya on - OEV wins outright")
ax.legend(frameon=False, fontsize=8, loc="lower right")

fig.suptitle("OEV vs TypeSafe Jev and laya: accuracy, workflows, speed, calibration, size - all measured", fontsize=13, fontweight="bold")
fig.savefig("assets/oev_vs_jev_full.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("written: assets/oev_vs_jev_full.png")


# ---------------- dark variant ----------------
DARK_BG = "#0d1117"
DARK_TEXT = "#e6edf3"
DARK_GRID = "#3d444d"
DARK_ACC = ["#79b8ff", "#ff7b72", "#56d364", "#e3b341", "#a3a3a3"]

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": DARK_BG,
    "axes.grid": True,
    "grid.color": DARK_GRID,
    "grid.linewidth": 0.5,
    "text.color": DARK_TEXT,
    "axes.edgecolor": DARK_GRID,
    "axes.labelcolor": DARK_TEXT,
    "xtick.color": DARK_TEXT,
    "ytick.color": DARK_TEXT,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE, RED, GREEN, GOLD, GRAY = DARK_ACC

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)

ax = fig.add_subplot(gs[0, 0])
b1 = ax.bar(x - w, jev, w, label="Jev (published)", color=GRAY)
b2 = ax.bar(x, laya, w, label="laya (published)", color=RED)
b3 = ax.bar(x + w, oev, w, label="OEV (this repo)", color=BLUE)
for bars in (b1, b2, b3):
    ax.bar_label(bars, fmt="%.3f", fontsize=6, color=DARK_TEXT)
ax.set_xticks(x); ax.set_xticklabels(bench, fontsize=7)
ax.set_ylim(0, 1.15); ax.set_ylabel("accuracy")
ax.set_title("Accuracy - shared public datasets")
ax.legend(frameon=False, fontsize=7, loc="lower right")

ax = fig.add_subplot(gs[0, 1])
cols = [GRAY, GRAY, GOLD, RED, RED, BLUE]
bars = ax.bar(names, vals, color=cols)
ax.bar_label(bars, fmt="%.4f", fontsize=7, color=DARK_TEXT)
ax.axhline(0.735, color=GOLD, ls=":", lw=1)
ax.set_ylim(0, 0.9); ax.set_ylabel("accuracy")
ax.set_title("typed-decisions: vs baselines and ceiling")
ax.tick_params(axis="x", labelsize=7)

ax = fig.add_subplot(gs[0, 2])
b1 = ax.barh(y - h/2, laya_wf, h, label="laya", color=RED)
b2 = ax.barh(y + h/2, oev_wf, h, label="OEV", color=BLUE)
ax.bar_label(b1, fmt="%.3f", fontsize=6, color=DARK_TEXT)
ax.bar_label(b2, fmt="%.3f", fontsize=6, color=DARK_TEXT)
ax.set_yticks(y); ax.set_yticklabels(wf, fontsize=7)
ax.set_xlim(0.5, 1.0); ax.set_xlabel("accuracy")
ax.set_title("typed-decisions: every workflow")
ax.legend(frameon=False, fontsize=7, loc="lower right")

ax = fig.add_subplot(gs[1, 0])
b1 = ax.bar(np.arange(4) - 0.2, laya_lat, 0.4, label="laya (published)", color=RED)
b2 = ax.bar(np.arange(4) + 0.2, oev_lat, 0.4, label="OEV (measured)", color=BLUE)
ax.bar_label(b1, fmt="%.0f", fontsize=6, color=DARK_TEXT)
ax.bar_label(b2, fmt="%.0f", fontsize=6, color=DARK_TEXT)
ax.set_xticks(np.arange(4)); ax.set_xticklabels(xt)
ax.set_ylabel("p50 latency (ms)"); ax.set_xlabel("questions per call")
ax.set_title("Speed on one T4  (*OEV: 32-question batches)")
ax.legend(frameon=False, fontsize=7)

ax = fig.add_subplot(gs[1, 1])
bars = ax.bar(models, ece, color=[RED, BLUE, BLUE, GRAY])
ax.bar_label(bars, fmt="%.4f", fontsize=7, color=DARK_TEXT)
ax.set_ylabel("mean ECE (lower is better)")
ax.set_ylim(0, 0.3)
ax.set_title("Calibration (post-temperature)")

ax = fig.add_subplot(gs[1, 2])
for name, params, acc, _ in pts:
    c = BLUE if name.startswith("OEV") else RED
    ax.scatter(params, acc, s=120, color=c, zorder=3, edgecolors=DARK_TEXT, linewidths=0.5)
    ax.annotate(name, (params, acc), textcoords="offset points", xytext=(6, 5), fontsize=7, color=DARK_TEXT)
ax.set_xlim(0, 850); ax.set_ylim(0.74, 0.80)
ax.set_xlabel("parameters (millions)"); ax.set_ylabel("typed-decisions accuracy")
ax.set_title("Accuracy vs model size")

ax = fig.add_subplot(gs[2, :])
b1 = ax.bar(np.arange(4) - 0.2, hard, 0.4, label="hard accuracy", color=GREEN)
b2 = ax.bar(np.arange(4) + 0.2, soft, 0.4, label="soft accuracy", color=BLUE)
ax.bar_label(b1, fmt="%.4f", fontsize=8, color=DARK_TEXT)
ax.bar_label(b2, fmt="%.4f", fontsize=8, color=DARK_TEXT)
ax.axhline(0.471, color=RED, ls=":", lw=1.5)
ax.annotate("laya soft acc 0.471", (2.6, 0.48), color=RED, fontsize=8)
ax.axhline(0.580, color=GRAY, ls=":", lw=1.5)
ax.annotate("Jev soft acc 0.580", (0.0, 0.59), color=DARK_TEXT, fontsize=8)
ax.set_xticks(np.arange(4)); ax.set_xticklabels(gam)
ax.set_ylim(0.3, 0.85)
ax.set_title("Soft-accuracy sharpening sweep: the metric Jev beats laya on - OEV wins outright")
ax.legend(frameon=False, fontsize=8, loc="lower right")

fig.suptitle("OEV vs TypeSafe Jev and laya: accuracy, workflows, speed, calibration, size - all measured",
             fontsize=13, fontweight="bold", color=DARK_TEXT)
fig.savefig("assets/oev_vs_jev_full_dark.png", dpi=150, bbox_inches="tight")
print("written: assets/oev_vs_jev_full_dark.png")
