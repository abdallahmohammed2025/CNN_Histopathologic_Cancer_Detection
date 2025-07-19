# 🧬 Histopathologic Cancer Detection – Mini Project (Kaggle)

This project was developed as part of a weekly machine learning mini-project to detect metastatic cancer in histopathologic image patches. The task is a **binary image classification problem** hosted on [Kaggle](https://www.kaggle.com/competitions/histopathologic-cancer-detection).

---

## 📌 Problem Description

The goal of this challenge is to **predict the presence of metastatic cancer** in the central 32x32 pixels of 96x96px histopathologic image patches from lymph node sections.

- Binary labels: `1` = tumor present, `0` = no tumor
- Challenge: very small regions contain the relevant features, despite large input images
- Evaluation metric: **AUC ROC**

---

## 📁 Dataset Overview

- **Total training images:** ~220,000
- **Image shape:** 96x96 pixels, RGB
- **Label file:** `train_labels.csv`
- **Test set:** Unlabeled, to be submitted via CSV

Each image is named by its hash ID, and images are stored in `train/` and `test/` directories. The **label** only reflects the center region (32x32 px), not the full patch.

---

## 📊 Exploratory Data Analysis (EDA)

Our EDA process includes:
- Class distribution imbalance visualization (tumor vs non-tumor)
- Sample image grid plots to visually assess differences
- Channel-wise pixel intensity histograms
- Data checks (duplicates, missing labels, corrupted files)

Based on EDA insights, we applied:
- Stratified sampling
- Balanced data loaders
- Normalization using ImageNet means/stds

---

## 🧠 Model Architectures

We trained and compared multiple CNN architectures:
- ✅ **Baseline CNN:** 3 conv layers + FC head
- ✅ **Transfer Learning:** Pretrained ResNet18
- ✅ **EfficientNetB0 (optional)**

Each model was evaluated using:
- Train/validation AUC ROC
- Training loss curves
- Confusion matrix
- ROC curves

We used cross-entropy loss and Adam optimizer. For tuning:
- Learning rate search
- Data augmentation: flips, rotations
- Early stopping

---

## 🧪 Results & Evaluation

| Model        | Validation AUC | Notes                        |
|--------------|----------------|------------------------------|
| Simple CNN   | 0.85           | From-scratch, no pretraining |
| ResNet18     | 0.96           | Transfer learning            |
| EfficientNet | *optional*     | TBD                          |

We observed:
- ResNet18 significantly outperformed custom CNN
- Data augmentation improved generalization
- Small batch sizes stabilized training

---

## 🧾 Conclusion

- Transfer learning was crucial due to data size and complexity
- AUC is a suitable metric due to class imbalance
- Future improvements:
  - Better augmentations (e.g., CutMix, MixUp)
  - Label smoothing
  - Test-time augmentation

---

## 📂 Project Structure

