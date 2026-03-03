# Feature-Importance

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-FF6F00?style=flat&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/status-active-brightgreen)

> A systematic feature blurring study to assess and rank the importance of **14 protein features** used in deep learning-based contact prediction.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Method](#method)
- [Features Analyzed](#features-analyzed)
- [File Structure](#file-structure)

---

## 🔬 Overview

This project evaluates the contribution of individual protein features by **blurring each feature** with its corresponding noise matrix and measuring the resulting drop in prediction accuracy. Features that cause the greatest accuracy drop when blurred are ranked as most important.

The large model uses **1022 channels** total. 14 key features are individually masked to assess their importance.

---

## ⚙️ Method

For each feature:
1. Replace the feature values with the **mean matrix** (repeated 441 times)
2. Run the model with the blurred feature as input
3. Record the prediction accuracy
4. Rank features by accuracy drop

---

## 📊 Features Analyzed

| Feature | Description |
|---------|-------------|
| COV441 | Covariance matrix (441 channels) |
| PDNET55 | PDNet 55-channel feature |
| Entropy | Entropy-based feature |
| PSSM | Position-specific scoring matrix |
| SA | Solvent accessibility |
| SS | Secondary structure |
| trROS | trRosetta-derived features |

---

## 📁 File Structure

```
Feature-Importance/
├── dataio_cov441blur.py       # Covariance blurring data loader
├── dataio_pdnet55.py          # PDNet55 data loader
├── dataio_pdnet_entropy.py    # Entropy feature loader
├── dataio_pdnet_pssm.py       # PSSM feature loader
├── dataio_pdnet_sa.py         # Solvent accessibility loader
├── dataio_pdnet_ss.py         # Secondary structure loader
├── dataio_ss_pdnet.py         # SS+PDNet combined loader
├── dataio_trROSblur.py        # trRosetta blurring loader
├── generator.py               # Data generator
├── metrics.py                 # Evaluation metrics
├── models.py                  # Model architecture
├── plots.py                   # Result visualization
└── train.all.py               # Training script
```
