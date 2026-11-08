import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

labels = ["Original\nCenterCrop\n(Baseline)", "Horizontal\nFlip", "Rotation\n+8°", "Rotation\n−8°",
          "Brightness\nUp", "Contrast\nUp", "TTA Ensemble\n(all 6)"]
acc  = [87.61, 89.38, 83.19, 86.73, 90.27, 88.50, 88.50]
prec = [84.69, 85.39, 79.83, 82.98, 86.06, 85.52, 84.50]
rec  = [86.66, 88.91, 83.11, 86.27, 89.89, 88.51, 88.51]
f1   = [85.12, 86.91, 81.02, 84.26, 87.74, 86.65, 86.21]
series = [("Accuracy", acc, "#1f4e79"), ("Macro precision", prec, "#7fa8c9"),
          ("Macro recall", rec, "#e08a1e"), ("Macro F1", f1, "#2e7d32")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(labels)); w = 0.2
ax.axvspan(len(labels)-1.5, len(labels)-0.5, color="#e8f1e4", zorder=0)
for k, (name, vals, c) in enumerate(series):
    xs = x + (k - 1.5) * w
    ax.bar(xs, vals, w, color=c, edgecolor="black", linewidth=0.5, label=name, zorder=3)
    for xi, v in zip(xs, vals):
        ax.text(xi, v + 0.25, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
ax.axhline(f1[0], color="#2e7d32", linestyle=":", linewidth=1, zorder=2)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.get_xticklabels()[-1].set_fontweight("bold")
ax.set_ylim(75, 94); ax.set_ylabel("Performance (%)")
ax.set_title("Individual TTA transforms versus the six-view TTA ensemble", fontsize=12, fontweight="bold", pad=12)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
h, l = ax.get_legend_handles_labels()
ax.legend(h + [Patch(facecolor="#e8f1e4", edgecolor="gray")], l + ["Ensemble"], loc="upper left", frameon=False, ncol=5)
fig.text(0.01, 0.005, "Dotted line: baseline macro-F1. y-axis starts at 75%. Held-out test set, n = 113.", fontsize=8, color="dimgray")
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("tta_chart.png", dpi=300, bbox_inches="tight", facecolor="white")
