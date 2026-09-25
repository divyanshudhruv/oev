"""Generate README charts from measured OEV numbers into assets/.

Benchmark numbers below are measured or published; decision primitive values are illustrative:
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




def _headline_scorecard(path, text, edge, sub, face):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6),
                             gridspec_kw={"wspace": 0.34})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.20)
    charts = [
        ("Typed decisions", ["single", "ensemble"], [0.7705, 0.7760],
         "accuracy", (0, 0.82), [P_BLUE, P_GREEN]),
        ("Banking77", ["soup"], [0.8584], "accuracy", (0, 0.95), [P_SAND]),
        ("Latency", ["T4", "CPU"], [22.2, 447], "ms per question (log scale)",
         (1, 700), [P_BLUE, P_RED]),
    ]
    for ax, (title, labels, values, ylabel, ylim, colors) in zip(axes, charts):
        bottom = 1 if "log scale" in ylabel else 0
        bars = ax.bar(labels, [v - bottom for v in values], bottom=bottom,
                      color=colors, width=0.56, edgecolor=edge, linewidth=0.5)
        ax.bar_label(bars, labels=[f"{v:.4f}" if v < 1 else f"{v:.1f} ms"
                                   for v in values], padding=4, color=text,
                      fontsize=9)
        ax.set_title(title, color=text, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, color=text)
        ax.set_ylim(*ylim)
        if "log scale" in ylabel:
            ax.set_yscale("log")
        ax.set_facecolor(face)
        ax.grid(axis="y", color=edge, alpha=0.25)
        ax.tick_params(colors=text)
        for spine in ax.spines.values():
            spine.set_color(edge)
    fig.suptitle("OEV headline results", color=text, fontsize=16, fontweight="bold", y=0.95)
    fig.text(0.07, 0.04, "Selected public results. See BENCHMARKS.md for protocol and caveats.",
             color=sub, fontsize=9)
    fig.savefig(path, dpi=150, transparent=True)
    plt.close(fig)


def _decision_primitives(path, text, edge, sub, face):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.subplots_adjust(left=0.19, right=0.97, top=0.82, bottom=0.19)
    rows = [
        ("choice", [("billing", 0.72, P_BLUE), ("support", 0.18, P_RED), ("other", 0.10, P_GREEN)]),
        ("noul", [("true", 0.86, P_BLUE), ("false", 0.14, P_RED)]),
        ("score", [(str(i), value, color) for i, value, color in zip(
            range(1, 6), [0.05, 0.10, 0.25, 0.35, 0.25], ACC)]),
    ]
    for y, (name, segments) in zip([2, 1, 0], rows):
        left = 0
        for label, value, color in segments:
            ax.barh(y, value, left=left, height=0.48, color=color,
                    edgecolor=edge, linewidth=0.5)
            ax.text(left + value / 2, y, f"{label} {value:.0%}", ha="center",
                    va="center", color=text, fontsize=8.5)
            left += value
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["choice", "noul", "score"], color=text, fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xticklabels([f"{int(value * 100)}%" for value in np.arange(0, 1.01, 0.2)])
    ax.set_xlabel("normalized output", color=text)
    ax.set_title("One packed sequence, three decision primitives", color=text,
                 fontweight="bold", pad=14)
    ax.set_facecolor(face)
    ax.grid(axis="x", color=edge, alpha=0.25)
    ax.tick_params(colors=text)
    for spine in ax.spines.values():
        spine.set_color(edge)
    fig.text(0.19, 0.04, "Illustrative distributions; choice, boolean, and ordered outputs share one mechanism.",
             color=sub, fontsize=9)
    fig.savefig(path, dpi=150, transparent=True)
    plt.close(fig)


def _latency_profile(path, text, edge, sub, face):
    fig, (gpu, cpu) = plt.subplots(1, 2, figsize=(10, 4.8),
                                   gridspec_kw={"width_ratios": [1.25, 0.75]})
    fig.subplots_adjust(wspace=0.32, left=0.10, right=0.96, top=0.78, bottom=0.19)
    gpu_labels = ["single", "batch 32"]
    gpu_values = [22.2, 15.9]
    bars = gpu.bar(gpu_labels, gpu_values, color=[P_BLUE, P_GREEN], width=0.56)
    gpu.bar_label(bars, labels=[f"{v:.1f} ms" for v in gpu_values], padding=4,
                  color=text, fontsize=10)
    gpu.set_title("GPU | Tesla T4, fp32", color=text, fontweight="bold")
    gpu.set_ylabel("ms per question", color=text)
    gpu.set_ylim(0, 27)
    cpu_bars = cpu.bar(["CPU"], [447], color=P_SAND, width=0.46)
    cpu.bar_label(cpu_bars, labels=["447 ms"], padding=4, color=text, fontsize=10)
    cpu.set_title("CPU | 8 threads", color=text, fontweight="bold")
    cpu.set_ylabel("ms per question", color=text)
    cpu.set_ylim(0, 500)
    for ax in (gpu, cpu):
        ax.set_facecolor(face)
        ax.grid(axis="y", color=edge, alpha=0.25)
        ax.tick_params(colors=text)
        for spine in ax.spines.values():
            spine.set_color(edge)
    fig.suptitle("Measured latency, hardware separated", color=text, fontsize=16,
                 fontweight="bold", y=0.95)
    fig.text(0.10, 0.04, "Batch value is per-question throughput; CPU timing is single-question p50.",
             color=sub, fontsize=9)
    fig.savefig(path, dpi=150, transparent=True)
    plt.close(fig)


_light_text, _light_edge, _light_sub, _light_face = "#24292f", "#57606a", "#6e7480", "#f6efe7"
_headline_scorecard("assets/headline_scorecard.png", _light_text, _light_edge, _light_sub, _light_face)
_decision_primitives("assets/decision_primitives.png", _light_text, _light_edge, _light_sub, _light_face)
_latency_profile("assets/latency_profile.png", _light_text, _light_edge, _light_sub, _light_face)

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

print("light charts written: benchmarks, zeroshot, transfer_speed, "
      "headline_scorecard, decision_primitives, latency_profile, workflows")

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




_dark_face = "#2b211c"
_headline_scorecard("assets/headline_scorecard_dark.png", DARK_TEXT, DARK_EDGE, DARK_SUB, _dark_face)
_decision_primitives("assets/decision_primitives_dark.png", DARK_TEXT, DARK_EDGE, DARK_SUB, _dark_face)
_latency_profile("assets/latency_profile_dark.png", DARK_TEXT, DARK_EDGE, DARK_SUB, _dark_face)


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
