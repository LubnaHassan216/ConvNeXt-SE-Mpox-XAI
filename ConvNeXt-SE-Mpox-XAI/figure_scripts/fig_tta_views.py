"""Figure: the six TTA views used at inference, for one sample image per selected class.
Usage (Colab):  python fig_tta_views.py /content/drive/MyDrive/FOLDS/Fold_1/test
Picks the first image of each class in CLASSES from <root>/<class>/ and saves Fig_TTA_views.png/.pdf
Uses exactly the evaluation pipeline of the paper: Resize(256x256) -> CenterCrop(224) -> view."""
import sys
from pathlib import Path
from PIL import Image
try:
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    base_fn = None
except ImportError:          # PIL fallback with the same operations
    from PIL import ImageOps, ImageEnhance
    T = None
    class TF:
        hflip = staticmethod(ImageOps.mirror)
        rotate = staticmethod(lambda im, a: im.rotate(a))
        adjust_brightness = staticmethod(lambda im, f: ImageEnhance.Brightness(im).enhance(f))
        adjust_contrast = staticmethod(lambda im, f: ImageEnhance.Contrast(im).enhance(f))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42})

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/content/drive/MyDrive/FOLDS/Fold_1/test")
CLASSES = ["Chickenpox", "Cowpox", "HFMD", "Healthy", "Measles", "Monkeypox"]   # all six classes
RESIZE, CROP = 256, 224                            # same values as the paper's eval transform

if T is not None:
    base = T.Compose([T.Resize((RESIZE, RESIZE)), T.CenterCrop((CROP, CROP))])
else:
    o = (RESIZE - CROP) // 2
    base = lambda im: im.resize((RESIZE, RESIZE), Image.BILINEAR).crop((o, o, o + CROP, o + CROP))
views = {
    "(a) Original\ncentre crop": lambda im: im,
    "(b) Horizontal\nflip":       TF.hflip,
    "(c) Rotation\n+8°":          lambda im: TF.rotate(im, 8),
    "(d) Rotation\n−8°":          lambda im: TF.rotate(im, -8),
    "(e) Brightness\n×1.08":      lambda im: TF.adjust_brightness(im, 1.08),
    "(f) Contrast\n×1.08":        lambda im: TF.adjust_contrast(im, 1.08),
}
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
samples = []
for c in CLASSES:
    files = sorted(p for p in (ROOT / c).iterdir() if p.suffix.lower() in exts)
    samples.append((c, Image.open(files[0]).convert("RGB")))

fig, axes = plt.subplots(len(samples), len(views), figsize=(1.9*len(views), 1.95*len(samples)))
if len(samples) == 1: axes = [axes]
for r, (c, im) in enumerate(samples):
    im = base(im)
    for k, (name, fn) in enumerate(views.items()):
        ax = axes[r][k]; ax.imshow(fn(im)); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_edgecolor("#2e7d32" if k == 0 else "#bdbdbd"); s.set_linewidth(1.2)
        if r == 0: ax.set_title(name, fontsize=9.5)
        if k == 0: ax.set_ylabel(c, fontsize=10, fontweight="bold")
fig.tight_layout(w_pad=0.4, h_pad=0.6)
fig.savefig("Fig_TTA_views.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig("Fig_TTA_views.pdf", bbox_inches="tight", facecolor="white")
print("saved Fig_TTA_views.png / .pdf")
