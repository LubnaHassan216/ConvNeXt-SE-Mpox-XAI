import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

FROZEN = "#d9e2ec"     # frozen in both stages
FT2 = "#f6c58f"        # trainable in Stage 2 only
TRAIN = "#a8d5a2"      # trainable in both stages
EDGE = "#33475b"
TXT = "#1b2631"

fig = plt.figure(figsize=(16, 8.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 84); ax.axis("off")


def box(x, y, w, h, color, title, sub=None, fs=9.5, bold=True, r=1.2, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=color, ec=EDGE, lw=lw, zorder=3))
    cy = y + h / 2 + (1.4 if sub else 0)
    ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else None, color=TXT, zorder=4)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 2.2, sub, ha="center", va="center",
                fontsize=fs - 1.6, color="#34495e", zorder=4, linespacing=1.25)


def arrow(x1, y1, x2, y2, style="-|>", lw=1.4, color=EDGE, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                                 lw=lw, color=color, ls=ls, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))


def panel(x, y, w, h, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2",
                                fc="#fbfcfd", ec="#9aa8b5", lw=1.0, ls=(0, (4, 3)), zorder=1))
    ax.text(x + 1.5, y + h - 2.0, label, ha="left", va="center", fontsize=11,
            fontweight="bold", color=TXT, zorder=4)


# ---------------------------------------------------------------- (a) backbone
panel(2, 44, 156, 38, "(a) ConvNeXt-Base backbone (ImageNet-1K pre-trained; features[0]–features[7])")

# input
ax.add_patch(Rectangle((5, 52), 11, 18, fc="#eef3f7", ec=EDGE, lw=1.1, zorder=3))
for k in range(5):
    ax.plot([6.5 + 2 * k, 6.5 + 2 * k], [54, 68], color="#c4d0db", lw=0.8, zorder=4)
ax.text(10.5, 61, "Input\nimage", ha="center", va="center", fontsize=9.5, fontweight="bold", zorder=5)
ax.text(10.5, 49.5, "224 × 224 × 3", ha="center", fontsize=8.5, color="#34495e")

mods = [  # x, w, h, colour, title, sub, below
    (19, 11, 18, FROZEN, "Stem", "4×4 conv, s4\n+ LayerNorm", "[0]  56×56×128"),
    (32.5, 13, 18, FROZEN, "Stage 1", "3 × ConvNeXt\nblocks", "[1]  56×56×128"),
    (48, 9, 18, FROZEN, "Down-\nsample", None, "[2]  28×28×256"),
    (59.5, 13, 18, FROZEN, "Stage 2", "3 × ConvNeXt\nblocks", "[3]  28×28×256"),
    (75, 9, 18, FROZEN, "Down-\nsample", None, "[4]  14×14×512"),
    (86.5, 14, 18, FROZEN, "Stage 3", "27 × ConvNeXt\nblocks", "[5]  14×14×512"),
    (103, 9, 18, FT2, "Down-\nsample", None, "[6]  7×7×1024"),
    (114.5, 14, 18, FT2, "Stage 4", "3 × ConvNeXt\nblocks", "[7]  7×7×1024"),
]
prev = 16
for x, w, h, c, t, s, b in mods:
    arrow(prev, 61, x, 61)
    box(x, 52, w, h, c, t, s, fs=9.5)
    ax.text(x + w / 2, 49.5, b, ha="center", fontsize=8.2, color="#34495e")
    prev = x + w

# ConvNeXt block detail
bx, by = 132, 50
ax.add_patch(FancyBboxPatch((bx, by), 24, 22, boxstyle="round,pad=0,rounding_size=1.5",
                            fc="white", ec="#9aa8b5", lw=0.9, zorder=3))
ax.text(bx + 12, by + 19.6, "ConvNeXt block", ha="center", fontsize=9, fontweight="bold", zorder=4)
for i, t in enumerate(["7×7 depthwise conv", "LayerNorm", "1×1 conv (4C) + GELU", "1×1 conv (C)", "+ residual"]):
    ax.text(bx + 12, by + 16 - 3.2 * i, t, ha="center", fontsize=8, color="#34495e", zorder=4)
arrow(121.5, 70, bx, 66, style="-", lw=0.8, color="#9aa8b5", ls=(0, (3, 2)))

ax.text(66, 75.6, "frozen during both training stages", ha="center", fontsize=8.8,
        color="#5d6d7e", style="italic")
ax.annotate("", xy=(103, 74.2), xytext=(128.5, 74.2),
            arrowprops=dict(arrowstyle="<->", color="#c0711c", lw=1.2))
ax.text(115.75, 75.8, "unfrozen in training Stage 2", ha="center", fontsize=8.8,
        color="#c0711c", fontweight="bold")

# ---------------------------------------------------------------- (b) SE + head
panel(2, 2, 156, 39, "(b) Squeeze-and-Excitation attention and classification head (trainable in both stages)")

# connector from Stage 4 down to SE
ax.add_patch(FancyArrowPatch((121.5, 52), (121.5, 45.5), arrowstyle="-", lw=1.4, color=EDGE, zorder=2))
ax.add_patch(FancyArrowPatch((121.5, 45.5), (8, 45.5), arrowstyle="-", lw=1.4, color=EDGE, zorder=2))
arrow(8, 45.5, 8, 33)
ax.text(64, 46.6, r"feature map  $X \in \mathbb{R}^{7\times7\times1024}$", ha="center", fontsize=8.8, color="#34495e")

# SE block
y0 = 20
box(3, y0, 10, 12, "#eef3f7", "X", "7×7×1024", fs=10)
se = [
    (16.5, 10, "GAP", "squeeze\n1024"),
    (29, 11, "FC + ReLU", "1024 → 64"),
    (42.5, 11, "FC +\nSigmoid", "64 → 1024"),
]
prev = 13
for x, w, t, s in se:
    arrow(prev, y0 + 6, x, y0 + 6)
    box(x, y0, w, 12, TRAIN, t, s, fs=9)
    prev = x + w
# scale
ax.add_patch(plt.Circle((60, y0 + 6), 2.4, fc="white", ec=EDGE, lw=1.2, zorder=3))
ax.text(60, y0 + 6, "×", ha="center", va="center", fontsize=13, fontweight="bold", zorder=4)
arrow(prev, y0 + 6, 57.6, y0 + 6)
ax.text(55.2, y0 + 7.2, "s", fontsize=9, style="italic", color="#34495e")
# skip from X to scale
ax.add_patch(FancyArrowPatch((8, y0), (8, 14), arrowstyle="-", lw=1.1, color=EDGE, zorder=2))
ax.add_patch(FancyArrowPatch((8, 14), (60, 14), arrowstyle="-", lw=1.1, color=EDGE, zorder=2))
arrow(60, 14, 60, y0 + 3.6, lw=1.1)
ax.text(34, 11.8, r"channel-wise recalibration:  $\tilde{X}_c = s_c \cdot X_c$", ha="center", fontsize=8.6, color="#34495e")
ax.add_patch(FancyBboxPatch((14.8, 17.2), 49.5, 19.3, boxstyle="round,pad=0,rounding_size=1.5",
                            fc="none", ec="#4f8a4b", lw=1.0, ls=(0, (3, 2)), zorder=2))
ax.text(39.5, 35.2, "SE block  (r = 16, 131,072 parameters)", ha="center", fontsize=8.8,
        color="#2e6b2a", fontweight="bold")

# GAP after SE
arrow(62.4, y0 + 6, 68, y0 + 6)
box(68, y0, 10, 12, "#eef3f7", "GAP", "1024", fs=9.5)
ax.text(121.5, 79.2, "Grad-CAM target layer: last block of Stage 4", ha="center", fontsize=8.4,
        color="#8e44ad", fontweight="bold")

# head
head = ["Layer-\nNorm", "Dropout\n0.35", "Linear\n1024\n+ GELU", "Dropout\n0.30", "Linear\n512\n+ GELU", "Dropout\n0.25", "Linear\n6"]
hx = 81.5
ax.add_patch(FancyBboxPatch((hx - 1, 17.2), 59.3, 19.3, boxstyle="round,pad=0,rounding_size=1.5",
                            fc="none", ec="#4f8a4b", lw=1.0, ls=(0, (3, 2)), zorder=2))
ax.text(hx + 28.6, 35.2, "Custom classification head", ha="center", fontsize=8.8,
        color="#2e6b2a", fontweight="bold")
arrow(78, y0 + 6, hx, y0 + 6)
w = 7.6
for i, t in enumerate(head):
    x = hx + i * 8.4
    box(x, y0, w, 12, TRAIN if "Dropout" not in t else "#d7ecd4", t, fs=7.6, bold="Dropout" not in t, r=0.9)
    if i < len(head) - 1:
        arrow(x + w, y0 + 6, x + 8.4, y0 + 6, lw=1.0)

# softmax + output
sx = hx + 6 * 8.4 + w
arrow(sx, y0 + 6, 143, y0 + 6)
box(143, y0, 13, 12, "#eef3f7", "Softmax", None, fs=9.5)
classes = ["Chickenpox", "Cowpox", "HFMD", "Healthy", "Measles", "Monkeypox"]
ax.text(149.5, 16.3, "\n".join(classes), ha="center", va="top", fontsize=7.8, color=TXT, linespacing=1.3)
ax.text(149.5, 17.9, "6 classes", ha="center", fontsize=8, fontweight="bold", color=TXT)

# loss note
ax.text(110, 8.5, "Loss: class-weighted cross-entropy with label smoothing (ε = 0.05)",
        ha="center", fontsize=8.6, color="#34495e", style="italic")

# legend
lx, ly = 4, 4.2
for c, t in [(FROZEN, "Frozen (both training stages)"), (FT2, "Fine-tuned in Stage 2"),
             (TRAIN, "Trained in Stages 1 and 2")]:
    ax.add_patch(FancyBboxPatch((lx, ly), 3.4, 2.4, boxstyle="round,pad=0,rounding_size=0.4",
                                fc=c, ec=EDGE, lw=0.8, zorder=3))
    ax.text(lx + 4.4, ly + 1.2, t, va="center", fontsize=8.6, color=TXT)
    lx += 27

fig.savefig("Figure3_ConvNeXt-SE_architecture.png", dpi=600, facecolor="white")
fig.savefig("Figure3_ConvNeXt-SE_architecture.pdf", facecolor="white")
print("saved")
