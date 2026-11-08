"""
ConvNeXt-SE: An Explainable Deep Transfer Learning Framework for
Multiclass Mpox-Related Skin Lesion Classification (MSLD v2.0, six classes)

Full implementation of the pipeline described in the paper:
  Sec. 3.1  Data loading (original clinical images only, exact-duplicate removal)
  Sec. 3.2  Online data augmentation (training split only)
  Sec. 3.3  ConvNeXt-Base backbone + Squeeze-and-Excitation block + custom head
  Sec. 3.4  Class-weighted cross-entropy + label smoothing; two-stage fine-tuning
  Sec. 3.5  Six-view Test-Time Augmentation (TTA) ensemble
  Sec. 4.2  Accuracy, macro precision / recall / F1, per-class and macro ROC-AUC
  Sec. 5.1  Repeated stratified 5-fold cross-validation (3 repeats = 15 folds)
  Sec. 5.3  Independent held-out test evaluation (70/15/15 split)
  Sec. 5.4  Cumulative ablation study + individual TTA views
  Sec. 5.5  Grad-CAM visual explanations

Usage (Google Colab):
  !pip install -q grad-cam
  !python convnext_se_mpox.py --data /content/drive/MyDrive/FOLDS/Fold_1 --mode all
Modes: cv | holdout | ablation | gradcam | all
--data may be either
  (a) a folder with train/ val/ test/ sub-folders, each containing one folder per class
      (all splits are pooled and re-split by this script), or
  (b) a folder containing one sub-folder per class.
All outputs are written to --out (default: ./outputs).
"""

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.transforms.functional as TF

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize


# ============================================================================
# Configuration (Table 5 of the paper)
# ============================================================================
CFG = dict(
    img_size=224,            # network input
    resize=256,              # resize before crop
    batch_size=16,
    num_workers=2,
    stage1_epochs=5,         # Stage 1: SE block + head only
    stage2_epochs=8,         # Stage 2: + last downsampling layer and Stage 4
    stage1_lr=5e-4,
    stage2_lr=5e-5,
    weight_decay=1e-4,
    patience=3,              # early stopping (validation macro-F1)
    plateau_patience=2,      # ReduceLROnPlateau
    plateau_factor=0.5,
    grad_clip=1.0,
    label_smoothing=0.05,    # epsilon
    se_reduction=16,         # r
    train_repeat_factor=2,   # each training image sampled twice per epoch
    n_splits=5,
    n_repeats=3,
    global_seed=42,
    test_frac=0.15,          # held-out split 70/15/15
    val_frac=0.15,
)

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Sec. 3.1  Dataset: original clinical images, exact duplicates removed
# ============================================================================
def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_dataset(root):
    """Return (paths, labels, class_names). Pools train/val/test if present and
    skips exact duplicates (identical MD5 hash)."""
    root = Path(root)
    split_dirs = [root / s for s in ("train", "val", "test") if (root / s).is_dir()]
    sources = split_dirs if split_dirs else [root]
    class_names = sorted({p.name for s in sources for p in s.iterdir() if p.is_dir()})
    c2i = {c: i for i, c in enumerate(class_names)}

    paths, labels, seen, dup = [], [], set(), 0
    for src in sources:
        for c in class_names:
            d = src / c
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() not in VALID_EXTS:
                    continue
                h = md5(f)
                if h in seen:
                    dup += 1
                    continue
                seen.add(h)
                paths.append(str(f))
                labels.append(c2i[c])
    paths, labels = np.array(paths), np.array(labels)
    counts = {c: int((labels == i).sum()) for i, c in enumerate(class_names)}
    print(f"Images: {len(paths)} | exact duplicates skipped: {dup}")
    print("Per class:", counts)
    return paths, labels, class_names


# ============================================================================
# Sec. 3.2  Online augmentation (Table 3) and evaluation transform
# ============================================================================
train_tf = transforms.Compose([
    transforms.Resize((CFG["resize"], CFG["resize"])),
    transforms.RandomResizedCrop(CFG["img_size"], scale=(0.75, 1.0), ratio=(0.90, 1.10)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08), shear=8),
    transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.06, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_base = transforms.Compose([
    transforms.Resize((CFG["resize"], CFG["resize"])),
    transforms.CenterCrop((CFG["img_size"], CFG["img_size"])),
])
to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
eval_tf = transforms.Compose([eval_base, to_tensor])


class SkinDataset(Dataset):
    def __init__(self, paths, labels, transform, repeat_factor=1):
        self.paths, self.labels = list(paths), list(labels)
        self.transform, self.repeat = transform, max(1, int(repeat_factor))

    def __len__(self):
        return len(self.paths) * self.repeat

    def __getitem__(self, idx):
        i = idx % len(self.paths)
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), torch.tensor(int(self.labels[i]), dtype=torch.long)


def make_loader(paths, labels, transform, shuffle, repeat=1):
    return DataLoader(SkinDataset(paths, labels, transform, repeat),
                      batch_size=CFG["batch_size"], shuffle=shuffle,
                      num_workers=CFG["num_workers"], pin_memory=(DEVICE == "cuda"))


# ============================================================================
# Sec. 3.3  ConvNeXt-SE architecture
# ============================================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation (Eqs. 1-3): GAP -> FC(C->C/r) -> ReLU -> FC(C/r->C) -> Sigmoid."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * s


class ClassificationHead(nn.Module):
    """Custom head (Table 4)."""
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(in_features), nn.Dropout(0.35),
            nn.Linear(in_features, 1024), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(512, num_classes))

    def forward(self, x):
        return self.net(x)


class ConvNeXtSE(nn.Module):
    """ConvNeXt-Base features -> (SE) -> GAP (Eq. 4) -> head -> logits (Softmax, Eq. 5)."""
    def __init__(self, num_classes, use_se=True, reduction=16, pretrained=True):
        super().__init__()
        weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.convnext_base(weights=weights)
        self.features = backbone.features          # 8 sequential modules
        self.attention = SEBlock(1024, reduction) if use_se else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = ClassificationHead(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.avgpool(x)
        return self.classifier(x)


def set_stage(model, stage):
    """Stage 1: freeze backbone, train SE + head.
    Stage 2: additionally unfreeze features[-2:] (last downsampling layer + Stage 4)."""
    for p in model.features.parameters():
        p.requires_grad = False
    if stage == 2:
        for m in list(model.features)[-2:]:
            for p in m.parameters():
                p.requires_grad = True
    for p in model.attention.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


# ============================================================================
# Sec. 3.4  Loss: class-weighted cross-entropy (Eqs. 6-7) + label smoothing (Eq. 8)
# ============================================================================
def class_weights(train_labels, num_classes):
    counts = np.bincount(train_labels, minlength=num_classes).astype(float)
    w = counts.sum() / (num_classes * counts)      # w_c = N / (C * n_c)
    w = w / w.mean()                               # normalised by the mean
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def make_criterion(train_labels, num_classes, weighted=True, smoothing=True):
    return nn.CrossEntropyLoss(
        weight=class_weights(train_labels, num_classes) if weighted else None,
        label_smoothing=CFG["label_smoothing"] if smoothing else 0.0)


# ============================================================================
# Training / evaluation
# ============================================================================
def macro_scores(y, p):
    acc = accuracy_score(y, p)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return acc, pr, rc, f1


def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    loss_sum, ys, ps = 0.0, [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * x.size(0)
        ys.extend(y.cpu().numpy()); ps.extend(out.argmax(1).cpu().numpy())
    return loss_sum / len(loader.dataset), *macro_scores(ys, ps)


@torch.no_grad()
def predict(model, loader, criterion=None):
    model.eval()
    loss_sum, ys, probs = 0.0, [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            out = model(x)
            if criterion is not None:
                loss_sum += criterion(out, y).item() * x.size(0)
        probs.append(torch.softmax(out.float(), 1).cpu().numpy())
        ys.extend(y.cpu().numpy())
    probs = np.concatenate(probs)
    return np.array(ys), probs, loss_sum / len(loader.dataset)


def run_stage(model, stage, train_loader, val_loader, criterion, history, tag):
    set_stage(model, stage)
    lr = CFG["stage1_lr"] if stage == 1 else CFG["stage2_lr"]
    epochs = CFG["stage1_epochs"] if stage == 1 else CFG["stage2_epochs"]
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=lr, weight_decay=CFG["weight_decay"])
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=CFG["plateau_factor"], patience=CFG["plateau_patience"])
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_f1, best_state, wait = -1.0, copy.deepcopy(model.state_dict()), 0
    for ep in range(1, epochs + 1):
        tr = train_epoch(model, train_loader, criterion, opt, scaler)
        y, pr, vloss = predict(model, val_loader, criterion)
        va = macro_scores(y, pr.argmax(1))
        sched.step(va[3])
        history.append(dict(tag=tag, stage=stage, epoch=ep, lr=opt.param_groups[0]["lr"],
                            train_loss=tr[0], train_acc=tr[1], train_f1=tr[4],
                            val_loss=vloss, val_acc=va[0], val_f1=va[3]))
        print(f"  [{tag}] S{stage} ep{ep:02d} | train loss {tr[0]:.4f} acc {tr[1]:.4f} F1 {tr[4]:.4f}"
              f" || val loss {vloss:.4f} acc {va[0]:.4f} F1 {va[3]:.4f}")
        if va[3] > best_f1:
            best_f1, best_state, wait = va[3], copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= CFG["patience"]:
                print(f"  early stopping at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return best_f1


def train_model(tr_p, tr_y, va_p, va_y, num_classes, use_se=True, weighted=True,
                smoothing=True, pretrained=True, history=None, tag=""):
    """Two-stage progressive fine-tuning (Sec. 3.4.2)."""
    history = [] if history is None else history
    model = ConvNeXtSE(num_classes, use_se, CFG["se_reduction"], pretrained).to(DEVICE)
    criterion = make_criterion(tr_y, num_classes, weighted, smoothing)
    tl = make_loader(tr_p, tr_y, train_tf, True, CFG["train_repeat_factor"])
    vl = make_loader(va_p, va_y, eval_tf, False)
    run_stage(model, 1, tl, vl, criterion, history, tag)
    run_stage(model, 2, tl, vl, criterion, history, tag)
    return model, criterion


# ============================================================================
# Sec. 3.5  Six-view Test-Time Augmentation (Eq. 9)
# ============================================================================
TTA_VIEWS = {
    "Original_CenterCrop": lambda im: im,
    "Horizontal_Flip": TF.hflip,
    "Rotation_Positive_8": lambda im: TF.rotate(im, 8),
    "Rotation_Negative_8": lambda im: TF.rotate(im, -8),
    "Brightness_Up": lambda im: TF.adjust_brightness(im, 1.08),
    "Contrast_Up": lambda im: TF.adjust_contrast(im, 1.08),
}


def view_transform(fn):
    return transforms.Compose([eval_base, transforms.Lambda(fn), to_tensor])


def evaluate_tta(model, paths, labels):
    """Returns per-view probabilities and the averaged (ensemble) probabilities."""
    per_view = {}
    for name, fn in TTA_VIEWS.items():
        y, probs, _ = predict(model, make_loader(paths, labels, view_transform(fn), False))
        per_view[name] = probs
    ensemble = np.mean(list(per_view.values()), axis=0)   # Eq. (9)
    return y, per_view, ensemble


# ============================================================================
# Reporting helpers
# ============================================================================
def full_metrics(y, probs, class_names):
    pred = probs.argmax(1)
    acc, pr, rc, f1 = macro_scores(y, pred)
    yb = label_binarize(y, classes=list(range(len(class_names))))
    per_auc = {c: float(roc_auc_score(yb[:, i], probs[:, i])) for i, c in enumerate(class_names)}
    macro_auc = float(roc_auc_score(yb, probs, average="macro", multi_class="ovr"))
    return dict(accuracy=acc, precision=pr, recall=rc, f1=f1,
                macro_auc=macro_auc, per_class_auc=per_auc)


def pct(d):
    return {k: (round(100 * v, 2) if isinstance(v, float) else v) for k, v in d.items()}


def plot_confusion(y, pred, class_names, path, title):
    cm = confusion_matrix(y, pred, labels=list(range(len(class_names))))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    fontsize=8, color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout()
    fig.savefig(path, dpi=300); plt.close(fig)


def plot_roc(y, probs, class_names, path):
    yb = label_binarize(y, classes=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, c in enumerate(class_names):
        fpr, tpr, _ = roc_curve(yb[:, i], probs[:, i])
        ax.plot(fpr, tpr, label=f"{c} (AUC = {100 * roc_auc_score(yb[:, i], probs[:, i]):.2f}%)")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    macro = roc_auc_score(yb, probs, average="macro", multi_class="ovr")
    ax.set_title(f"ROC curves (macro AUC = {100 * macro:.2f}%)")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8); fig.tight_layout()
    fig.savefig(path, dpi=300); plt.close(fig)


def holdout_split(paths, labels, seed):
    """Stratified 70/15/15 split."""
    tv_p, te_p, tv_y, te_y = train_test_split(
        paths, labels, test_size=CFG["test_frac"], stratify=labels, random_state=seed)
    rel_val = CFG["val_frac"] / (1 - CFG["test_frac"])
    tr_p, va_p, tr_y, va_y = train_test_split(
        tv_p, tv_y, test_size=rel_val, stratify=tv_y, random_state=seed)
    return tr_p, tr_y, va_p, va_y, te_p, te_y


# ============================================================================
# Sec. 5.1  Repeated stratified 5-fold cross-validation (3 x 5 = 15 folds)
# ============================================================================
def run_cv(paths, labels, class_names, out, pretrained=True):
    rows, history = [], []
    for r in range(1, CFG["n_repeats"] + 1):
        skf = StratifiedKFold(CFG["n_splits"], shuffle=True, random_state=CFG["global_seed"] + r)
        for f, (tr, va) in enumerate(skf.split(paths, labels), 1):
            set_seed(CFG["global_seed"] + 100 * r + f)
            print(f"\n=== CV repeat {r} fold {f} ===")
            model, crit = train_model(paths[tr], labels[tr], paths[va], labels[va],
                                      len(class_names), pretrained=pretrained,
                                      history=history, tag=f"R{r}F{f}")
            y, probs, loss = predict(model, make_loader(paths[va], labels[va], eval_tf, False), crit)
            acc, pr, rc, f1 = macro_scores(y, probs.argmax(1))
            p_c, r_c, f1_c, s_c = precision_recall_fscore_support(
                y, probs.argmax(1), labels=list(range(len(class_names))), zero_division=0)
            row = dict(repeat=r, fold=f, val_loss=loss, accuracy=acc,
                       precision=pr, recall=rc, f1=f1)
            for i, c in enumerate(class_names):
                row.update({f"{c}_precision": p_c[i], f"{c}_recall": r_c[i],
                            f"{c}_f1": f1_c[i], f"{c}_support": int(s_c[i])})
            rows.append(row)
            print(f"  fold result: acc {acc:.4f} | P {pr:.4f} | R {rc:.4f} | F1 {f1:.4f}")
            del model; torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(out / "cv_fold_results.csv", index=False)
    pd.DataFrame(history).to_csv(out / "cv_training_history.csv", index=False)

    n = len(df)
    summary = []
    for m in ["accuracy", "precision", "recall", "f1"]:
        mu, sd = df[m].mean(), df[m].std(ddof=1)
        h = 1.96 * sd / np.sqrt(n)
        summary.append(dict(metric=m, mean=100 * mu, std=100 * sd,
                            ci_low=100 * (mu - h), ci_high=100 * (mu + h)))
    summary = pd.DataFrame(summary).round(2)
    summary.to_csv(out / "cv_summary_table6.csv", index=False)

    per_class = pd.DataFrame([dict(
        cls=c,
        precision=100 * df[f"{c}_precision"].mean(),
        recall=100 * df[f"{c}_recall"].mean(),
        f1=100 * df[f"{c}_f1"].mean(),
        support=df[f"{c}_support"].mean()) for c in class_names]).round(1)
    per_class.to_csv(out / "cv_per_class_table7.csv", index=False)

    # Figure: macro-F1 per fold
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(range(1, n + 1), df["f1"], marker="o", label="Fold macro-F1")
    ax.axhline(df["f1"].mean(), linestyle="--", label=f"Mean = {df['f1'].mean():.3f}")
    ax.set_xlabel("Fold index across repeats (1-5: R1, 6-10: R2, 11-15: R3)")
    ax.set_ylabel("Validation macro F1-score"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "cv_macro_f1_per_fold.png", dpi=300); plt.close(fig)

    print("\nTABLE 6 - CV summary (%):\n", summary.to_string(index=False))
    print("\nTABLE 7 - per-class CV (%):\n", per_class.to_string(index=False))
    return summary


# ============================================================================
# Sec. 5.3  Held-out test evaluation (+ TTA, ROC-AUC, Table 11)
# ============================================================================
def run_holdout(paths, labels, class_names, out, seed, pretrained=True):
    set_seed(seed)
    tr_p, tr_y, va_p, va_y, te_p, te_y = holdout_split(paths, labels, seed)
    print(f"\nHeld-out split (seed {seed}): train {len(tr_p)} | val {len(va_p)} | test {len(te_p)}")
    history = []
    model, crit = train_model(tr_p, tr_y, va_p, va_y, len(class_names),
                              pretrained=pretrained, history=history, tag=f"holdout_s{seed}")
    pd.DataFrame(history).to_csv(out / f"holdout_s{seed}_history.csv", index=False)
    torch.save(model.state_dict(), out / f"convnext_se_seed{seed}.pth")

    # standard inference = the Original_CenterCrop view (identical transform)
    y, views, ens = evaluate_tta(model, te_p, te_y)
    std = full_metrics(y, views["Original_CenterCrop"], class_names)
    tta = full_metrics(y, ens, class_names)

    # Table 11: each view separately + ensemble
    rows = []
    for name, probs in list(views.items()) + [("TTA_Ensemble", ens)]:
        a, p, r, f = macro_scores(y, probs.argmax(1))
        rows.append(dict(view=name, accuracy=100 * a, precision=100 * p, recall=100 * r, f1=100 * f))
    tta_df = pd.DataFrame(rows).round(2)
    tta_df.to_csv(out / f"holdout_s{seed}_tta_views_table11.csv", index=False)

    report = classification_report(y, ens.argmax(1), target_names=class_names, digits=4, zero_division=0)
    (out / f"holdout_s{seed}_classification_report.txt").write_text(report)
    plot_confusion(y, ens.argmax(1), class_names, out / f"holdout_s{seed}_confusion_tta.png",
                   "TTA ensemble - normalized confusion matrix")
    plot_roc(y, ens, class_names, out / f"holdout_s{seed}_roc_tta.png")
    np.save(out / f"holdout_s{seed}_test_probs_tta.npy", ens)
    pd.DataFrame({"path": te_p, "label": te_y}).to_csv(out / f"holdout_s{seed}_test_split.csv", index=False)

    result = dict(seed=seed, standard=pct({k: v for k, v in std.items() if k != "per_class_auc"}),
                  tta=pct({k: v for k, v in tta.items() if k != "per_class_auc"}),
                  per_class_auc_tta={k: round(100 * v, 2) for k, v in tta["per_class_auc"].items()})
    (out / f"holdout_s{seed}_results.json").write_text(json.dumps(result, indent=2))

    print("\nTABLE 8 - standard vs TTA (%):")
    print(pd.DataFrame({"Standard": result["standard"], "TTA": result["tta"]}).to_string())
    print("\nTABLE 9 - per-class AUC (TTA, %):", result["per_class_auc_tta"])
    print("\nTABLE 11 - individual TTA views (%):\n", tta_df.to_string(index=False))
    print("\nClassification report (TTA ensemble):\n", report)
    return model, (te_p, te_y), result


# ============================================================================
# Sec. 5.4  Cumulative ablation (Table 10), averaged over seeds
# ============================================================================
ABLATION = [  # name, use_se, label_smoothing, weighted_loss, tta
    ("ConvNeXt baseline",     False, False, False, False),
    ("+ SE block",            True,  False, False, False),
    ("+ Label smoothing",     True,  True,  False, False),
    ("+ Weighted loss",       True,  True,  True,  False),
    ("+ TTA ensemble",        True,  True,  True,  True),
]


def run_ablation(paths, labels, class_names, out, seeds, pretrained=True):
    rows = []
    for seed in seeds:
        tr_p, tr_y, va_p, va_y, te_p, te_y = holdout_split(paths, labels, seed)
        trained = {}
        for name, se, ls, wl, tta in ABLATION:
            key = (se, ls, wl)
            if key not in trained:      # the TTA row re-uses the full model
                set_seed(seed)
                print(f"\n=== Ablation | seed {seed} | {name} ===")
                trained[key], _ = train_model(tr_p, tr_y, va_p, va_y, len(class_names),
                                              use_se=se, weighted=wl, smoothing=ls,
                                              pretrained=pretrained, tag=f"abl_s{seed}")
            model = trained[key]
            if tta:
                y, _, probs = evaluate_tta(model, te_p, te_y)
            else:
                y, probs, _ = predict(model, make_loader(te_p, te_y, eval_tf, False))
            a, p, r, f = macro_scores(y, probs.argmax(1))
            rows.append(dict(seed=seed, model=name, accuracy=100 * a, f1=100 * f))
        del trained; torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation_all_seeds.csv", index=False)
    order = [a[0] for a in ABLATION]
    table = (df.groupby("model")[["accuracy", "f1"]].agg(["mean", "std"])
               .reindex(order).round(2))
    table.to_csv(out / "ablation_table10.csv")
    print("\nTABLE 10 - ablation (mean, std over seeds, %):\n", table.to_string())
    return table


# ============================================================================
# Sec. 5.5  Grad-CAM (Eqs. 16-17)
# ============================================================================
def run_gradcam(model, test_set, class_names, out, per_class=1):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("Grad-CAM skipped: install with  pip install grad-cam")
        return
    te_p, te_y = test_set
    model.eval()
    cam = GradCAM(model=model, target_layers=[model.features[-1][-1]])
    picks = []
    for c in range(len(class_names)):
        idx = np.where(te_y == c)[0][:per_class]
        picks.extend(idx.tolist())
    fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.6))
    axes = np.array(axes).reshape(2, -1)
    for k, i in enumerate(picks):
        img = eval_base(Image.open(te_p[i]).convert("RGB"))
        rgb = np.asarray(img, dtype=np.float32) / 255.0
        x = to_tensor(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = int(model(x).argmax(1))
        heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
        axes[0, k].imshow(rgb); axes[0, k].set_title(f"True: {class_names[te_y[i]]}", fontsize=9)
        axes[1, k].imshow(show_cam_on_image(rgb, heat, use_rgb=True))
        ok = "correct" if pred == te_y[i] else "wrong"
        axes[1, k].set_title(f"Pred: {class_names[pred]} ({ok})", fontsize=9,
                             color="green" if ok == "correct" else "red")
        for a in axes[:, k]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(out / "gradcam_examples.png", dpi=300); plt.close(fig)
    print("Saved Grad-CAM figure:", out / "gradcam_examples.png")


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="ConvNeXt-SE for MSLD v2.0")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--mode", default="all", choices=["cv", "holdout", "ablation", "gradcam", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                    help="seeds for held-out evaluation / ablation")
    ap.add_argument("--no-pretrained", action="store_true", help="debug only")
    ap.add_argument("--quick", action="store_true", help="debug: 1 epoch per stage, 1 repeat x 2 folds")
    args = ap.parse_args()

    if args.quick:
        CFG.update(stage1_epochs=1, stage2_epochs=1, n_repeats=1, n_splits=2,
                   batch_size=4, num_workers=0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(CFG, indent=2))
    set_seed(CFG["global_seed"])
    print("Device:", DEVICE)

    paths, labels, class_names = load_dataset(args.data)
    pretrained = not args.no_pretrained

    if args.mode in ("cv", "all"):
        run_cv(paths, labels, class_names, out, pretrained)

    model = test_set = None
    if args.mode in ("holdout", "gradcam", "all"):
        results = []
        for s in (args.seeds if args.mode != "gradcam" else args.seeds[:1]):
            m, ts, res = run_holdout(paths, labels, class_names, out, s, pretrained)
            results.append(res)
            if model is None:
                model, test_set = m, ts
            else:
                del m
        if len(results) > 1:
            agg = {}
            for part in ("standard", "tta"):
                for k in ("accuracy", "precision", "recall", "f1", "macro_auc"):
                    v = [r[part][k] for r in results]
                    agg[f"{part}_{k}"] = f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}"
            (out / "holdout_mean_std.json").write_text(json.dumps(agg, indent=2))
            print("\nHeld-out mean ± std over seeds:", json.dumps(agg, indent=2))

    if args.mode in ("ablation", "all"):
        run_ablation(paths, labels, class_names, out, args.seeds, pretrained)

    if args.mode in ("gradcam", "all") and model is not None:
        run_gradcam(model, test_set, class_names, out)

    print("\nAll outputs saved in:", out.resolve())


if __name__ == "__main__":
    main()
