# ConvNeXt-SE-Mpox-XAI

## ConvNeXt-SE: An Explainable Deep Transfer Learning Framework for Multiclass Mpox-Related Skin Lesion Classification

This repository contains the computational implementation associated with the study on multiclass classification of Mpox-related skin lesions using a ConvNeXt-Base backbone enhanced with a Squeeze-and-Excitation (SE) attention block, two-stage transfer learning, test-time augmentation (TTA) and explainable artificial intelligence (Grad-CAM).

## Repository Contents

### Dataset

The `Dataset` folder contains information about the image data used in this study.

The study uses the **Mpox Skin Lesion Dataset Version 2.0 (MSLD v2.0)**:

* **Classes (6):** Monkeypox, Chickenpox, Cowpox, Hand-Foot-and-Mouth Disease (HFMD), Measles, Healthy
* **Images:** 755 original clinical images 
* **Source:** Kaggle – Mpox Skin Lesion Dataset Version 2.0 (MSLD v2.0)
* **Reference:** S. N. Ali et al., *Biomedical Signal Processing and Control*, 2024, doi: 10.1016/j.bspc.2024.106742

Only the **original** images are used. The offline-augmented images distributed with the dataset are **not** used. The images are not redistributed in this repository and should be obtained from the original source.

### ConvNeXt-SE-Mpox-XAI

The `ConvNeXt-SE-Mpox-XAI` folder contains the main implementation (`convnext_se_mpox.py`), including:

* Data loading and exact-duplicate removal
* Online data augmentation (training data only)
* ConvNeXt-Base backbone with SE attention block and custom classification head
* Class-weighted cross-entropy loss with label smoothing
* Two-stage progressive fine-tuning
* Repeated stratified 5-fold cross-validation (3 repeats, 15 folds)
* Independent held-out test evaluation
* Six-view test-time augmentation (TTA) ensemble
* ROC-AUC analysis (per class and macro-averaged)
* Cumulative ablation study
* Grad-CAM visual explanations

The `figure_scripts` sub-folder contains the scripts used to produce the figures of the paper.

### Figures

The `Figures` folder contains the figures generated during the analysis and used to visualize model performance and explainability results.

## Methodology

The overall computational workflow consists of:

1. MSLD v2.0 data collection (original images only)
2. Exact-duplicate removal and stratified data partitioning
3. Online data augmentation of the training data
4. ConvNeXt-Base + SE attention + custom classification head
5. Class-weighted cross-entropy with label smoothing (ε = 0.05)
6. Two-stage progressive fine-tuning
7. Repeated stratified 5-fold cross-validation (3 × 5 folds)
8. Independent held-out test evaluation (70/15/15 split)
9. Six-view test-time augmentation ensemble
10. Ablation study
11. Grad-CAM explainability analysis

## Model

| Component | Setting |
| --- | --- |
| Backbone | ConvNeXt-Base, ImageNet-1K pre-trained (torchvision) |
| Attention | Squeeze-and-Excitation, 1024 channels, reduction ratio r = 16 |
| Classification head | LayerNorm → Dropout 0.35 → Linear 1024 → GELU → Dropout 0.30 → Linear 512 → GELU → Dropout 0.25 → Linear 6 |
| Stage 1 | Backbone frozen; SE block and head trained; 5 epochs; learning rate 5 × 10⁻⁴ |
| Stage 2 | Last downsampling layer and Stage 4 unfrozen; 8 epochs; learning rate 5 × 10⁻⁵ |
| Optimizer | AdamW, weight decay 1 × 10⁻⁴ |
| Scheduler | ReduceLROnPlateau on validation macro-F1 (factor 0.5, patience 2) |
| Early stopping | Patience 3 (validation macro-F1) |
| Batch size / input | 16 / 224 × 224 |

## Test-Time Augmentation

At inference, six deterministic views of each test image are generated (original centre crop, horizontal flip, rotation +8°, rotation −8°, brightness ×1.08, contrast ×1.08). The softmax probabilities of the six views are averaged to obtain the final prediction.

## Explainable AI

Grad-CAM (Gradient-weighted Class Activation Mapping) is applied to the last block of the ConvNeXt backbone to visualize the image regions that contribute most to each prediction.

## Reproducibility

The code was run on Google Colab Pro+ with an NVIDIA A100 GPU.

```bash
pip install -r requirements.txt
python ConvNeXt-SE-Mpox-XAI/convnext_se_mpox.py --data <path_to_MSLD_v2.0> --mode all
```

`--data` can point either to a folder with one sub-folder per class, or to a folder with `train/`, `val/` and `test/` sub-folders (these are pooled and re-split by the script).

Available modes: `cv`, `holdout`, `ablation`, `gradcam`, `all`. All results (tables as CSV/JSON, figures and trained weights) are written to `outputs/`. Random seeds are fixed (global seed 42; held-out seeds 42, 43, 44).

## Code Availability

The code is provided to support reproducibility of the computational analysis and can be accessed by the Editorial Board and peer reviewers during the peer-review process.

## Data Source

Mpox Skin Lesion Dataset Version 2.0 (MSLD v2.0), Kaggle:

https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20


