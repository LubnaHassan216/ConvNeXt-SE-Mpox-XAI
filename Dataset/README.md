# Dataset

The image data used in this study were obtained from the **Mpox Skin Lesion Dataset Version 2.0 (MSLD v2.0)**.

**Classes:** Monkeypox, Chickenpox, Cowpox, Hand-Foot-and-Mouth Disease (HFMD), Measles, Healthy

| Class | Original images |
| --- | --- |
| Monkeypox | 284 |
| Chickenpox | 75 |
| Cowpox | 66 |
| HFMD | 161 |
| Healthy | 114 |
| Measles | 55 |
| **Total** | **755** |

One exact duplicate (identical file content, detected by MD5 hash) is removed automatically by the code, leaving 754 unique images.

Original data source:

https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20

Dataset reference:

S. N. Ali, M. T. Ahmed, T. Jahan, J. Paul, S. M. S. Sani, N. Noor, A. N. Asma, and T. Hasan, "A web-based mpox skin lesion detection system using state-of-the-art deep learning models considering racial diversity," *Biomedical Signal Processing and Control*, 2024, doi: 10.1016/j.bspc.2024.106742.

Only the **original images** are used in this study; the offline-augmented images provided with the dataset are not used. The images are not redistributed here and should be downloaded directly from the original source.

Expected folder layout (either works):

```
MSLD_v2.0/                      MSLD_v2.0/
├── Chickenpox/                 ├── train/<class folders>
├── Cowpox/                     ├── val/<class folders>
├── HFMD/                       └── test/<class folders>
├── Healthy/
├── Measles/
└── Monkeypox/
```

Data loading, duplicate removal and data partitioning are described in the main script.
