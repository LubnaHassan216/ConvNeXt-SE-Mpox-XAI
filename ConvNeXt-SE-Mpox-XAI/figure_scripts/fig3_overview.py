"""Figure 3 (overview style): ConvNeXt-SE framework, four columns, corrected."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
fig = plt.figure(figsize=(17, 9.6))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 170); ax.set_ylim(0, 96); ax.axis("off")

COLS = [("#4a6585", "#eef2f7", "DATA PREPARATION\n& VALIDATION"),
        ("#3b3f8f", "#eeeff9", "CONVNEXT-BASE\nBACKBONE"),
        ("#1f8a8a", "#e8f5f4", "SE ATTENTION &\nCLASSIFICATION"),
        ("#d0782a", "#fdf3ea", "EVALUATION\n& XAI")]
X0, W, GAP = 3, 38, 4.0
TXT, SUB = "#1b2631", "#3d4f61"
FROZEN, FT2, TRAIN = "#d9e2ec", "#f6c58f", "#a8d5a2"


def rbox(x, y, w, h, fc, ec, lw=1.2, r=1.5, z=2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z, ls=ls))


def arrow(x1, y1, x2, y2, c="#33475b", lw=1.3, ms=11, z=5, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                                 lw=lw, color=c, zorder=z))


def t(x, y, s, fs=9, b=False, c=TXT, ha="center", va="center", it=False, z=6, ls=1.2):
    ax.text(x, y, s, fontsize=fs, fontweight="bold" if b else None, color=c, ha=ha, va=va,
            style="italic" if it else None, zorder=z, linespacing=ls)


def bullet(x, y, s):
    ax.add_patch(Polygon([[x, y - 0.8], [x, y + 0.8], [x + 1.2, y]], closed=True, fc=TXT, zorder=6))
    t(x + 2.2, y, s, fs=10.5, b=True, ha="left")


# ------------------------------------------------------------------ columns
xs = [X0 + i * (W + GAP) for i in range(4)]
for (hc, bg, title), x in zip(COLS, xs):
    rbox(x, 3, W, 90, bg, hc, lw=2.2, r=2.5, z=1)
    rbox(x, 80, W, 13, hc, hc, lw=2.2, r=2.5, z=2)
    ax.add_patch(Rectangle((x, 80), W, 3, fc=hc, ec="none", zorder=2))
    t(x + W / 2, 86.5, title, fs=14, b=True, c="white", ls=1.1)
# big arrows between columns
labels = ["ORIGINAL IMAGES\n(online aug.)", "FEATURE MAP\n7×7×1024", "CLASS\nPROBABILITIES"]
for i in range(3):
    xa = xs[i] + W - 1.0
    hc = COLS[i][0]
    ax.add_patch(Polygon([[xa, 50], [xa + 3.4, 50], [xa + 3.4, 47], [xa + 6.2, 53.5],
                          [xa + 3.4, 60], [xa + 3.4, 57], [xa, 57]], closed=True,
                         fc=hc, ec="white", lw=1.2, zorder=7))

# ================================================================== column 1
x = xs[0]
bullet(x + 2, 76, "MSLD v2.0 dataset (Kaggle)")
# database icon
for k in range(3):
    ax.add_patch(matplotlib.patches.Ellipse((x + 8, 65 + 3 * k), 8, 2.4, fc="#c9d6e3", ec="#4a6585", lw=1, zorder=4))
ax.add_patch(Rectangle((x + 4, 65), 8, 6, fc="#c9d6e3", ec="none", zorder=3))
ax.add_patch(matplotlib.patches.Ellipse((x + 8, 71), 8, 2.4, fc="#dbe5ef", ec="#4a6585", lw=1, zorder=5))
arrow(x + 13.5, 68, x + 17.5, 68, c="#4a6585")
rbox(x + 18, 62.5, 18, 11, "white", "#4a6585", lw=1)
t(x + 27, 69.8, "755 original images", fs=8.4, b=True)
t(x + 27, 66.2, "6 classes", fs=7.8, c=SUB)

bullet(x + 2, 58, "Preprocessing & augmentation")
rbox(x + 3, 43.5, 33, 11.5, "white", "#4a6585", lw=1)
t(x + 19.5, 52.5, "Resize 256 → crop 224 · ImageNet norm.", fs=7.6, c=SUB)
t(x + 19.5, 49.6, "Online aug. (training only): crop, flip,", fs=7.6, c=SUB)
t(x + 19.5, 47.3, "rotation ±15°, affine, colour jitter", fs=7.6, c=SUB)
t(x + 19.5, 45.0, "Class-weighted loss · label smoothing", fs=7.6, c=SUB)

bullet(x + 2, 39.5, "Repeated stratified 5-fold CV")
t(x + 19, 36.2, "3 repeats × 5 folds = 15 runs", fs=8.2, c=SUB, it=True)
for k in range(5):
    yy = 30.5 - k * 3.6
    for j in range(5):
        c = "#d9534f" if j == k else "#5b8db8"
        ax.add_patch(Rectangle((x + 6 + j * 4.6, yy), 4.4, 2.6, fc=c, ec="white", lw=0.8, zorder=4))
    t(x + 4.2, yy + 1.3, f"F{k+1}", fs=7.2, c=SUB)
ax.add_patch(Rectangle((x + 30.5, 27.6), 2.4, 2.0, fc="#5b8db8", zorder=4)); t(x + 34.6, 28.6, "train", fs=7.2, c=SUB, ha="left")
ax.add_patch(Rectangle((x + 30.5, 24.2), 2.4, 2.0, fc="#d9534f", zorder=4)); t(x + 34.6, 25.2, "val.", fs=7.2, c=SUB, ha="left")
rbox(x + 3, 5, 33, 8.0, "white", "#4a6585", lw=1)
t(x + 19.5, 10.6, "Independent held-out split", fs=8.4, b=True)
t(x + 19.5, 7.4, "70% train · 15% val. · 15% test (n = 113)", fs=7.6, c=SUB)

# ================================================================== column 2
x = xs[1]
rbox(x + 2, 22, 22, 56, "#f7f8fd", "#3b3f8f", lw=1.2)
t(x + 13, 75.3, "ConvNeXt-Base", fs=9.5, b=True)
t(x + 13, 72.9, "feature extractor (ImageNet-1K)", fs=7.2, c=SUB)
mods = [("Stem  4×4, s4", "56×56×128", FROZEN), ("Stage 1  (3 blocks)", "56×56×128", FROZEN),
        ("Downsample", "28×28×256", FROZEN), ("Stage 2  (3 blocks)", "28×28×256", FROZEN),
        ("Downsample", "14×14×512", FROZEN), ("Stage 3  (27 blocks)", "14×14×512", FROZEN),
        ("Downsample", "7×7×1024", FT2), ("Stage 4  (3 blocks)", "7×7×1024", FT2)]
yy = 66.5
for i, (name, dim, c) in enumerate(mods):
    h = 4.2 if "Stage" in name or "Stem" in name else 3.0
    rbox(x + 4, yy - h, 18, h, c, "#3b3f8f", lw=0.9, r=0.8, z=3)
    t(x + 13, yy - h / 2, name, fs=7.4 if h > 3.5 else 6.9, b="Stage" in name or "Stem" in name)
    t(x + 13, yy - h - 0.9, f"[{i}] {dim}", fs=6.0, c=SUB) if False else None
    if i < len(mods) - 1:
        arrow(x + 13, yy - h, x + 13, yy - h - 1.4, lw=0.9, ms=7)
    yy = yy - h - 1.4
t(x + 13, 23.6, "output: 7 × 7 × 1024", fs=7.4, c=SUB, it=True)
# side: 2-stage transfer learning
t(x + 31, 76, "TWO-STAGE", fs=10.5, b=True, c="#3b3f8f")
t(x + 31, 73, "TRANSFER", fs=10.5, b=True, c="#3b3f8f")
t(x + 31, 70, "LEARNING", fs=10.5, b=True, c="#3b3f8f")
# lock icons
def lock(cx, cy, open_=False, c="#3b3f8f"):
    rbox(cx - 2.2, cy - 2.0, 4.4, 3.4, c, c, lw=1, r=0.4, z=6)
    th = np.linspace(0, np.pi, 30)
    xs_ = cx + 1.4 * np.cos(th) + (1.2 if open_ else 0); ys_ = cy + 1.4 + 1.6 * np.sin(th)
    ax.plot(xs_, ys_, color=c, lw=2, zorder=6)
lock(x + 31, 62)
t(x + 31, 57.6, "Stage 1", fs=9, b=True)
t(x + 31, 54.8, "backbone frozen", fs=7.6, c=SUB)
t(x + 31, 52.6, "SE + head trained", fs=7.6, c=SUB)
t(x + 31, 50.4, "5 epochs · LR 5e-4", fs=7.4, c=SUB, it=True)
lock(x + 31, 44, open_=True, c="#d0782a")
t(x + 31, 39.6, "Stage 2", fs=9, b=True)
t(x + 31, 36.8, "unfreeze last", fs=7.6, c=SUB)
t(x + 31, 34.6, "downsampling", fs=7.6, c=SUB)
t(x + 31, 32.4, "layer + Stage 4", fs=7.6, c=SUB)
t(x + 31, 30.2, "8 epochs · LR 5e-5", fs=7.4, c=SUB, it=True)
# legend
for i, (c, s) in enumerate([(FROZEN, "frozen in both stages"), (FT2, "fine-tuned in Stage 2"),
                            (TRAIN, "trained in both stages")]):
    yy = 16.5 - i * 3.6
    rbox(x + 4, yy - 1.1, 3.2, 2.2, c, "#3b3f8f", lw=0.8, r=0.4, z=4)
    t(x + 8.6, yy, s, fs=8, c=SUB, ha="left")
t(x + 19, 5.6, "AdamW · ReduceLROnPlateau · early stopping", fs=7.2, c=SUB, it=True)

# ================================================================== column 3
x = xs[2]
rbox(x + 2, 52, 34, 25, "white", "#1f8a8a", lw=1.2)
t(x + 19, 74.5, "Squeeze-and-Excitation block", fs=9.5, b=True)
t(x + 19, 71.9, "r = 16 · 131,072 parameters", fs=7.4, c=SUB, it=True)
# feature cube
for k in range(4):
    ax.add_patch(Rectangle((x + 3.5 + k * 0.9, 57.5 + k * 0.9), 4, 8, fc="#2f6f8f", ec="white", lw=0.6, zorder=4))
t(x + 7.4, 68.6, "X", fs=8.5, b=True)
se = [("GAP", "1024", 10.5), ("FC+ReLU", "1024→64", 17.5), ("FC+Sigm.", "64→1024", 24.5)]
arrow(x + 8.6, 62, x + 10, 62, lw=0.9, ms=8)
for i, (a, b, xx) in enumerate(se):
    rbox(x + xx, 58.5, 6.2, 7.5, TRAIN, "#1f8a8a", lw=0.9, r=0.6, z=4)
    t(x + xx + 3.1, 63.4, a, fs=6.6, b=True)
    t(x + xx + 3.1, 60.6, b, fs=6.0, c=SUB)
    if i < 2:
        arrow(x + xx + 6.2, 62, x + xx + 7, 62, lw=0.9, ms=8)
ax.add_patch(Circle((x + 33.4, 62), 1.6, fc="white", ec="#1f8a8a", lw=1.2, zorder=5))
t(x + 33.4, 62, "×", fs=11, b=True)
arrow(x + 30.7, 62, x + 31.8, 62, lw=0.9, ms=8)
ax.plot([x + 7, x + 7, x + 33.4, x + 33.4], [57.4 + 3.6 - 3.6, 54.0, 54.0, 60.4], color="#1f8a8a", lw=1, zorder=3)
t(x + 20, 55.2, r"recalibration  $\tilde{X}_c = s_c \cdot X_c$", fs=7.4, c=SUB)
arrow(x + 19, 52, x + 19, 49.3, c="#1f8a8a", lw=1.6, ms=13)

rbox(x + 2, 37, 34, 12, "white", "#1f8a8a", lw=1.2)
t(x + 19, 46.6, "Global Average Pooling", fs=9.5, b=True)
for k in range(3):
    for j in range(3):
        ax.add_patch(Rectangle((x + 6 + j * 1.6, 39 + k * 1.6), 1.5, 1.5, fc="#2f6f8f", ec="white", lw=0.5, zorder=4))
arrow(x + 11.5, 41.4, x + 15, 41.4, lw=0.9, ms=8)
for j in range(8):
    ax.add_patch(Rectangle((x + 16 + j * 1.9, 40.6), 1.8, 1.6, fc=plt.cm.viridis(j / 8), ec="white", lw=0.4, zorder=4))
t(x + 23.6, 38.6, "vector  1 × 1024", fs=7.2, c=SUB)
arrow(x + 19, 37, x + 19, 34.3, c="#1f8a8a", lw=1.6, ms=13)

rbox(x + 2, 5, 34, 29, "white", "#1f8a8a", lw=1.2)
t(x + 19, 31.6, "Custom classification head", fs=9.5, b=True)
head = [("LayerNorm", TRAIN), ("Dropout 0.35", "#d7ecd4"), ("Linear 1024 + GELU", TRAIN),
        ("Dropout 0.30", "#d7ecd4"), ("Linear 512 + GELU", TRAIN), ("Dropout 0.25", "#d7ecd4"),
        ("Linear 6", TRAIN)]
yy = 28.6
for i, (s, c) in enumerate(head):
    rbox(x + 4, yy - 2.6, 17, 2.6, c, "#1f8a8a", lw=0.8, r=0.5, z=4)
    t(x + 12.5, yy - 1.3, s, fs=7.0, b="Dropout" not in s)
    yy -= 3.25
# output classes
classes = ["Chickenpox", "Cowpox", "HFMD", "Healthy", "Measles", "Monkeypox"]
arrow(x + 21.3, 16.5, x + 23.8, 16.5, lw=1, ms=9)
rbox(x + 24, 14.6, 10.6, 3.8, "#eef6f6", "#1f8a8a", lw=0.9, r=0.5, z=4)
t(x + 29.3, 16.5, "Softmax", fs=7.8, b=True)
for i, c in enumerate(classes):
    t(x + 29.3, 12.2 - i * 1.25, c, fs=6.4, c=SUB)
t(x + 29.3, 26.5, "6 classes", fs=8, b=True, c="#1f8a8a")
t(x + 29.3, 23.8, "weighted CE", fs=6.8, c=SUB)
t(x + 29.3, 21.9, "+ label smooth.", fs=6.8, c=SUB)
t(x + 29.3, 20.0, "ε = 0.05", fs=6.8, c=SUB)

# ================================================================== column 4
x = xs[3]
hc = "#d0782a"
t(x + 19, 76.5, "Evaluation pathways", fs=11, b=True)


def img_icon(cx, cy, s=1.0):
    ax.add_patch(Rectangle((cx - 2.2 * s, cy - 1.7 * s), 4.4 * s, 3.4 * s, fc="#dfe9f2", ec="#4a6585", lw=0.9, zorder=6))
    ax.add_patch(Polygon([[cx - 2.0 * s, cy - 1.5 * s], [cx - 0.5 * s, cy + 0.6 * s], [cx + 0.6 * s, cy - 0.5 * s],
                          [cx + 1.3 * s, cy + 0.2 * s], [cx + 2.0 * s, cy - 1.5 * s]], closed=True,
                         fc="#7aa37a", ec="none", zorder=7))
    ax.add_patch(Circle((cx + 1.1 * s, cy + 1.0 * s), 0.4 * s, fc="#e3b23c", zorder=7))


# single pass
rbox(x + 2, 62, 34, 11.5, "white", hc, lw=1.2)
t(x + 19, 71.2, "Standard (single-pass) inference", fs=8.6, b=True)
img_icon(x + 8, 66.2); arrow(x + 11, 66.2, x + 16, 66.2, lw=1, ms=9)
rbox(x + 16.5, 64.2, 9, 4, "#fdf3ea", hc, lw=0.9, r=0.5, z=5); t(x + 21, 66.2, "model", fs=7.4, b=True)
arrow(x + 25.5, 66.2, x + 28.5, 66.2, lw=1, ms=9); t(x + 32, 66.2, "ŷ", fs=10, b=True)

# TTA
rbox(x + 2, 35, 34, 25, "white", hc, lw=1.2)
t(x + 19, 57.7, "Test-time augmentation (6 views)", fs=8.6, b=True)
views = ["centre crop", "h-flip", "rot. +8°", "rot. −8°", "bright. ×1.08", "contr. ×1.08"]
img_icon(x + 6, 46.5)
for i, v in enumerate(views):
    vy = 54.0 - i * 3.0
    arrow(x + 8.4, 46.5, x + 11.3, vy, lw=0.7, ms=6)
    rbox(x + 11.5, vy - 1.15, 11.5, 2.3, "#fdf3ea", hc, lw=0.7, r=0.4, z=5)
    t(x + 17.25, vy, v, fs=6.5)
    arrow(x + 23, vy, x + 26.3, 46.5, lw=0.7, ms=6)
ax.add_patch(Circle((x + 28, 46.5), 1.7, fc=hc, ec="white", lw=1, zorder=6))
t(x + 28, 46.5, "Σ", fs=10, b=True, c="white", z=7)
arrow(x + 29.7, 46.5, x + 32, 46.5, lw=1, ms=9); t(x + 33.7, 46.5, "ŷ", fs=10, b=True)
t(x + 19, 37.2, "average of the 6 softmax outputs", fs=7.2, c=SUB, it=True)

# metrics
rbox(x + 2, 26, 34, 7, "white", hc, lw=1.2)
t(x + 19, 31.0, "Metrics", fs=8.6, b=True)
t(x + 19, 28.0, "accuracy · macro P / R / F1 · ROC-AUC", fs=7.4, c=SUB)

# Grad-CAM
rbox(x + 2, 5, 34, 19, "white", hc, lw=1.2)
t(x + 19, 21.6, "Grad-CAM explainability", fs=8.6, b=True)
t(x + 19, 19.2, "target: last block of Stage 4", fs=7.0, c=SUB, it=True)
rbox(x + 4, 11.5, 10, 4.5, "#fdf3ea", hc, lw=0.8, r=0.5, z=5); t(x + 9, 13.75, "gradients\n∂yᶜ/∂Aᵏ", fs=6.4)
rbox(x + 4, 6.0, 10, 4.5, "#fdf3ea", hc, lw=0.8, r=0.5, z=5); t(x + 9, 8.25, "feature\nmaps Aᵏ", fs=6.4)
arrow(x + 14, 13.7, x + 17.3, 11.6, lw=0.9, ms=8); arrow(x + 14, 8.3, x + 17.3, 10.4, lw=0.9, ms=8)
ax.add_patch(Circle((x + 19, 11), 1.8, fc="white", ec=hc, lw=1.1, zorder=6)); t(x + 19, 11, "ReLU\nΣ", fs=5.6, b=True)
arrow(x + 20.8, 11, x + 23.5, 11, lw=0.9, ms=8)
yy_, xx_ = np.mgrid[-1:1:60j, -1:1:60j]
heat = np.exp(-((xx_ - 0.15) ** 2 + (yy_ + 0.1) ** 2) / 0.18)
ax.imshow(heat, cmap="jet", extent=(x + 25, x + 33, 8.2, 16.2), zorder=6, alpha=0.9)
ax.add_patch(Rectangle((x + 25, 8.2), 8, 8, fc="none", ec="#4a6585", lw=0.9, zorder=7))
t(x + 29, 6.6, "class heatmap", fs=6.6, c=SUB)

fig.savefig("Figure3_overview.pdf", facecolor="white")
fig.savefig("Figure3_overview.png", dpi=450, facecolor="white")
print("saved")
