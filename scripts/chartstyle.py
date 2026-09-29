"""Shared chart style: one palette, rcParams, and light/dark text tones."""
import matplotlib.pyplot as plt

# pastel palette, nudged darker so blue/green separate clearly
P_BLUE, P_RED, P_GREEN, P_SAND, P_GRAY = "#8db4e4", "#ee9e9e", "#a3c794", "#e2cc8f", "#a5aeb8"
ACC = [P_BLUE, P_RED, P_GREEN, P_SAND, P_GRAY]

# light-page text tones (SUB: footer gray, GRAY_ANNOT: in-chart annotations)
TEXT, EDGE, SUB, LINE = "#24292f", "#57606a", "#6e7480", "#c8ccd0"
GRAY_ANNOT = "#8b949e"
# dark-page text tones
DARK_TEXT, DARK_SUB, DARK_EDGE, DARK_LINE = "#e6edf3", "#9aa4ae", "#6e7681", "#3d444d"
# panel faces used by the scorecard-style figures
LIGHT_FACE, DARK_FACE = "#f6efe7", "#2b211c"

BASE_RC = {
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.transparent": True,
    "axes.grid": True,
    "grid.color": LINE,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.6,
    "axes.titlepad": 14,
    "figure.autolayout": False,
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": TEXT,
    "axes.edgecolor": EDGE,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
}

DARK_RC = {
    "text.color": DARK_TEXT,
    "axes.edgecolor": DARK_EDGE,
    "axes.labelcolor": DARK_TEXT,
    "xtick.color": DARK_TEXT,
    "ytick.color": DARK_TEXT,
    "grid.color": DARK_LINE,
}


def apply_dark():
    # swap text/edge/grid tones for the dark variants (base stays light)
    plt.rcParams.update(DARK_RC)


def edge_bars(bars, color):
    # consistent hairline edge on bar patches in both variants
    for p in bars.patches:
        p.set_edgecolor(color)
        p.set_linewidth(0.4)
