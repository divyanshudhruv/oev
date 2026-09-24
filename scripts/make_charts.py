"""Generate README charts from measured OEV numbers into assets/."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("assets", exist_ok=True)

DARK = "#1a1a2e"
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

# ---------------- Chart 1: accuracy across benchmarks ----------------
# Kev publishes no in-domain numbers for these datasets; its published
# numbers are OOD-suite accuracies, so its bars are omitted here rather
# than compared apples-to-oranges.
bench = ["typed-decisions", "AG News", "emotion", "Banking77"]
laya_vals = [0.766, 0.950, 0.595, 0.425]
oev_vals = [0.7760, 0.9489, 0.9300, 0.8529]
jev_vals = [0.727, 0.910, 0.480, 0.870]

fig, ax = plt.subplots(figsize=(12, 6.2))
x = range(len(bench))
w = 0.26
b1 = ax.bar([i - w for i in x], jev_vals, w, label="Jev (closed API, published)", color=ACC[4])
b2 = ax.bar(list(x), laya_vals, w, label="laya (published)", color=ACC[1])
b3 = ax.bar([i + w for i in x], oev_vals, w, label="OEV (this repo)", color=ACC[0])
ax.set_xticks(list(x))
ax.set_xticklabels(bench)
ax.set_ylim(0, 1.18)
ax.set_ylabel("accuracy")
ax.set_title("OEV vs published numbers on shared public benchmarks", fontweight="bold", pad=15)
ax.legend(frameon=False, loc="upper left", fontsize=12)
ax.bar_label(b1, fmt="%.3f", fontsize=10, padding=3)
ax.bar_label(b2, fmt="%.3f", fontsize=10, padding=3)
ax.bar_label(b3, fmt="%.3f", fontsize=10, padding=3)
fig.tight_layout()
fig.savefig("assets/benchmarks.png", dpi=150)
plt.close(fig)

# ---------------- Chart 2: accuracy vs params ----------------
fig, ax = plt.subplots(figsize=(7.5, 4.8))
pts = [
    ("laya-typed-decisions", 421, 0.766, ACC[1]),
    ("laya", 421, 0.950, ACC[1]),
    ("OEV ensemble", 4 * 184, 0.7760, ACC[0]),
    ("OEV single", 184, 0.7705, ACC[0]),
    ("OEV base (AG News)", 184, 0.9489, ACC[0]),
    ("OEV base (emotion)", 184, 0.9300, ACC[0]),
    ("OEV b77 (ens)", 3 * 184, 0.8529, ACC[0]),
    ("OEV b77 single", 184, 0.8403, ACC[0]),
    ("Jev (API, size n/a)", 700, 0.727, ACC[4]),
    ("Kev-0.8B (OOD suite)", 800, 0.837, ACC[2]),
    ("Kev-4B (OOD suite)", 4000, 0.852, ACC[2]),
]
for name, params, acc, c in pts:
    ax.scatter(params, acc, s=130, color=c, zorder=3, edgecolors="black", linewidths=0.5)
    dx, dy = (6, 4) if "Jev" not in name else (6, -12)
    ax.annotate(name, (params, acc), textcoords="offset points", xytext=(dx, dy), fontsize=9)
ax.set_xlabel("parameters (millions)")
ax.set_ylabel("accuracy")
ax.set_title("accuracy vs model size", fontweight="bold")
ax.set_xlim(0, 4200)
fig.tight_layout()
fig.savefig("assets/params_vs_acc.png", dpi=150)
plt.close(fig)

# ---------------- Chart 3: sharpening sweep (soft acc vs gamma) ----------------
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

print("charts written: assets/benchmarks.png, assets/params_vs_acc.png, assets/sharpening.png")

# ---------------- Chart 4: per-workflow accuracy (typed-decisions) ----------------
fig, ax = plt.subplots(figsize=(9, 4.8))
workflows = ["invoice\nprocessing", "customer\nservice", "agent-trace\nobservability", "security\nincidents"]
laya_wf = [0.804, 0.764, 0.730, 0.766]
oev_wf = [0.8360, 0.8040, 0.7400, 0.7220]
x = range(len(workflows))
w = 0.34
b1 = ax.bar([i - w / 2 for i in x], laya_wf, w, label="laya (published)", color=ACC[1])
b2 = ax.bar([i + w / 2 for i in x], oev_wf, w, label="OEV (4-voter ensemble)", color=ACC[0])
ax.bar_label(b1, fmt="%.3f", fontsize=9, padding=2)
ax.bar_label(b2, fmt="%.3f", fontsize=9, padding=2)
ax.axhline(0.766, color=ACC[1], ls=":", lw=1)
ax.axhline(0.776, color=ACC[0], ls=":", lw=1)
ax.set_xticks(list(x))
ax.set_xticklabels(workflows)
ax.set_ylim(0.6, 0.9)
ax.set_ylabel("accuracy")
ax.set_title("typed-decisions per-workflow: OEV wins 3 of 4", fontweight="bold")
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig("assets/workflows.png", dpi=150)
plt.close(fig)

print("assets/workflows.png added")

# ---------------- dark-mode variants ----------------
DARK_BG = "#0d1117"
DARK_PANEL = "#161b22"
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

for tag, accent in (("dark", DARK_ACC),):
    # benchmarks
    fig, ax = plt.subplots(figsize=(12, 6.2))
    b1 = ax.bar([i - w for i in x], jev_vals, w, label="Jev (closed API, published)", color=accent[4])
    b2 = ax.bar(list(x), laya_vals, w, label="laya (published)", color=accent[1])
    b3 = ax.bar([i + w for i in x], oev_vals, w, label="OEV (this repo)", color=accent[0])
    ax.set_xticks(list(x))
    ax.set_xticklabels(bench)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("accuracy")
    ax.set_title("OEV vs published numbers on shared public benchmarks", fontweight="bold", pad=15)
    ax.legend(frameon=False, loc="upper left", fontsize=12)
    ax.bar_label(b1, fmt="%.3f", fontsize=10, padding=3, color=DARK_TEXT)
    ax.bar_label(b2, fmt="%.3f", fontsize=10, padding=3, color=DARK_TEXT)
    ax.bar_label(b3, fmt="%.3f", fontsize=10, padding=3, color=DARK_TEXT)
    fig.tight_layout()
    fig.savefig(f"assets/benchmarks_{tag}.png", dpi=150)
    plt.close(fig)

    # params vs acc
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, params, acc, _ in pts:
        c = accent[0] if name.startswith("OEV") else (accent[1] if "laya" in name else accent[4])
        ax.scatter(params, acc, s=130, color=c, zorder=3, edgecolors=DARK_TEXT, linewidths=0.5)
        dx, dy = (6, 4) if "Jev" not in name else (6, -12)
        ax.annotate(name, (params, acc), textcoords="offset points", xytext=(dx, dy), fontsize=9, color=DARK_TEXT)
    ax.set_xlabel("parameters (millions)")
    ax.set_ylabel("accuracy")
    ax.set_title("accuracy vs model size", fontweight="bold")
    ax.set_xlim(0, 4200)
    fig.tight_layout()
    fig.savefig(f"assets/params_vs_acc_{tag}.png", dpi=150)
    plt.close(fig)

    # sharpening
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(gammas, soft, "o-", color=accent[0], lw=2, label="soft accuracy")
    ax.plot(gammas, hard, "s--", color=accent[2], lw=2, label="hard accuracy")
    ax.axhline(0.471, color=accent[1], ls=":", lw=1.5)
    ax.annotate("laya soft acc 0.471", (1.05, 0.477), color=accent[1], fontsize=9)
    for g, sv in zip(gammas, soft):
        ax.annotate(f"{sv:.4f}", (g, sv), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center", color=DARK_TEXT)
    ax.set_xlabel("sharpening exponent gamma")
    ax.set_ylabel("accuracy")
    ax.set_title("post-hoc sharpening sweep (3-voter ensemble)", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"assets/sharpening_{tag}.png", dpi=150)
    plt.close(fig)

    # workflows
    fig, ax = plt.subplots(figsize=(9, 4.8))
    b1 = ax.bar([i - w / 2 for i in x], laya_wf, w, label="laya (published)", color=accent[1])
    b2 = ax.bar([i + w / 2 for i in x], oev_wf, w, label="OEV (4-voter ensemble)", color=accent[0])
    ax.bar_label(b1, fmt="%.3f", fontsize=9, padding=2, color=DARK_TEXT)
    ax.bar_label(b2, fmt="%.3f", fontsize=9, padding=2, color=DARK_TEXT)
    ax.axhline(0.766, color=accent[1], ls=":", lw=1)
    ax.axhline(0.776, color=accent[0], ls=":", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(workflows)
    ax.set_ylim(0.6, 0.9)
    ax.set_ylabel("accuracy")
    ax.set_title("typed-decisions per-workflow: OEV wins 3 of 4", fontweight="bold")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"assets/workflows_{tag}.png", dpi=150)
    plt.close(fig)

print("dark variants written")


# ---------------- Chart 5: three-up panel (params | primitive | workflow) ----------------# reset to the light theme: the dark-variant loop above overwrote rcParams globally
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": "black",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# panel 1: accuracy vs params
ax = axes[0]
for name, params, acc, c in pts:
    ax.scatter(params, acc, s=110, color=c, zorder=3, edgecolors="black", linewidths=0.5)
    ax.annotate(name.replace(" (this repo)", ""), (params, acc),
                textcoords="offset points", xytext=(5, 6), fontsize=8)
ax.set_xlabel("parameters (M)"); ax.set_ylabel("typed-decisions accuracy")
ax.set_title("accuracy vs size")
ax.set_xlim(0, 4200)

# panel 2: per-primitive
ax = axes[1]
prim = ["noul", "choice", "score"]
oev_p = [0.8533, 0.7483, 0.7375]
laya_p = [0.857, 0.733, 0.723]
x2 = range(3); w2 = 0.35
b1 = ax.bar([i - w2/2 for i in x2], laya_p, w2, label="laya", color=ACC[1])
b2 = ax.bar([i + w2/2 for i in x2], oev_p, w2, label="OEV", color=ACC[0])
ax.bar_label(b1, fmt="%.3f", fontsize=8)
ax.bar_label(b2, fmt="%.3f", fontsize=8)
ax.set_xticks(list(x2)); ax.set_xticklabels(prim)
ax.set_ylim(0.6, 0.95); ax.set_ylabel("accuracy")
ax.set_title("per primitive")
ax.legend(frameon=False, fontsize=9)

# panel 3: per-workflow
ax = axes[2]
wf2 = ["invoice", "customer", "agent-trace", "security"]
oev_w = [0.8360, 0.8040, 0.7400, 0.7220]
laya_w = [0.804, 0.764, 0.730, 0.766]
x3 = range(4)
b1 = ax.bar([i - w2/2 for i in x3], laya_w, w2, label="laya", color=ACC[1])
b2 = ax.bar([i + w2/2 for i in x3], oev_w, w2, label="OEV", color=ACC[0])
ax.bar_label(b1, fmt="%.2f", fontsize=7)
ax.bar_label(b2, fmt="%.2f", fontsize=7)
ax.set_xticks(list(x3)); ax.set_xticklabels(wf2, fontsize=8, rotation=15)
ax.set_ylim(0.6, 0.95); ax.set_ylabel("accuracy")
ax.set_title("per workflow")
ax.legend(frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig("assets/breakdown.png", dpi=150)
plt.close(fig)

# dark variant
plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": DARK_BG,
    "text.color": DARK_TEXT, "axes.edgecolor": DARK_GRID,
    "axes.labelcolor": DARK_TEXT, "xtick.color": DARK_TEXT, "ytick.color": DARK_TEXT,
})
BLUE, RED = DARK_ACC[0], DARK_ACC[1]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
ax = axes[0]
for name, params, acc, _ in pts:
    c = DARK_ACC[0] if name.startswith("OEV") else DARK_ACC[1]
    ax.scatter(params, acc, s=110, color=c, zorder=3, edgecolors=DARK_TEXT, linewidths=0.5)
    ax.annotate(name.replace(" (this repo)", ""), (params, acc),
                textcoords="offset points", xytext=(5, 6), fontsize=8, color=DARK_TEXT)
ax.set_xlabel("parameters (M)"); ax.set_ylabel("typed-decisions accuracy")
ax.set_title("accuracy vs size"); ax.set_xlim(0, 4200)

ax = axes[1]
b1 = ax.bar([i - w2/2 for i in x2], laya_p, w2, label="laya", color=RED)
b2 = ax.bar([i + w2/2 for i in x2], oev_p, w2, label="OEV", color=BLUE)
ax.bar_label(b1, fmt="%.3f", fontsize=8, color=DARK_TEXT)
ax.bar_label(b2, fmt="%.3f", fontsize=8, color=DARK_TEXT)
ax.set_xticks(list(x2)); ax.set_xticklabels(prim)
ax.set_ylim(0.6, 0.95); ax.set_ylabel("accuracy")
ax.set_title("per primitive"); ax.legend(frameon=False, fontsize=9)

ax = axes[2]
b1 = ax.bar([i - w2/2 for i in x3], laya_w, w2, label="laya", color=RED)
b2 = ax.bar([i + w2/2 for i in x3], oev_w, w2, label="OEV", color=BLUE)
ax.bar_label(b1, fmt="%.2f", fontsize=7, color=DARK_TEXT)
ax.bar_label(b2, fmt="%.2f", fontsize=7, color=DARK_TEXT)
ax.set_xticks(list(x3)); ax.set_xticklabels(wf2, fontsize=8, rotation=15)
ax.set_ylim(0.6, 0.95); ax.set_ylabel("accuracy")
ax.set_title("per workflow"); ax.legend(frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig("assets/breakdown_dark.png", dpi=150)
plt.close(fig)

print("assets/breakdown.png + breakdown_dark.png written")
