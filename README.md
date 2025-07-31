# 🧬 Histopathologic Cancer Detection - Deep Learning Mini Project

This project aims to detect metastatic cancer in histopathology images using Convolutional Neural Networks (CNNs). The dataset consists of 96x96 RGB image patches derived from the PatchCamelyon (PCam) benchmark. Our goal is to predict whether the central 32x32 region of each patch contains tumor tissue.

---

## 📁 Dataset

- **Source**: [Kaggle - Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection)
- **Input shape**: 96x96 RGB patches
- **Labels**: `1` = tumor, `0` = normal
- **Balanced dataset**: Equal distribution of tumor and normal classes

---

## 🔍 Exploratory Data Analysis (EDA)

- Sample images of both classes (tumor and non-tumor) displayed
- **Histograms** for:
  - Class distribution
  - Pixel intensity (RGB channels)
- **Image similarity**:
  - Cosine similarity metric used on pixel vectors
- All images confirmed to be the same shape (96x96x3)
- Pixel value ranges across images show consistent distribution

---

## 🧠 Model Architectures

We experimented with the following CNN architectures:

1. **Simple CNN**:
   - Baseline model with a few convolutional + pooling layers
   - Overfit easily with limited regularization

2. **EfficientNetB0**:
   - Pretrained on ImageNet
   - Chosen for its balance between accuracy and efficiency
   - Fine-tuned on top layers only

3. **ResNet50**:
   - Deeper architecture with skip connections
   - Higher memory and training time, but strong feature extraction

---

## ⚙️ Hyperparameter Tuning

| Parameter         | Tried Values             | Best Value  |
|------------------|--------------------------|-------------|
| Learning Rate     | `1e-2`, `1e-3`, `1e-4`    | `1e-4`      |
| Batch Size        | `32`, `64`, `128`         | `64`        |
| Dropout Rate      | `0.2`, `0.4`, `0.5`        | `0.4`       |
| Pretrained Model  | `None`, `EffNetB0`, `ResNet50` | `EffNetB0` |

- **Why some worked better**:
  - EfficientNetB0 had fewer parameters and faster convergence
  - Pretrained weights gave a strong head-start
- **What didn’t work well**:
  - Simple CNN overfit without aggressive regularization
  - ResNet50 was slow and required more tuning

---

## 📊 Results

| Model          | ROC AUC | Accuracy | Parameters | Training Time |
|----------------|---------|----------|------------|----------------|
| Simple CNN     | 0.81    | 75%      | 500K       | Fast           |
| ResNet50       | 0.91    | 86%      | 23M        | Slow           |
| EfficientNetB0 | **0.94** | **89%**  | 5.3M       | Moderate       |

- Best model: **EfficientNetB0**
- Evaluation Metric: **ROC AUC** due to class balancing needs

---

## 🛠️ Troubleshooting

- **Low validation performance**:
  - Re-checked label balance and image preprocessing
  - Verified thresholding and activation function behavior

- **Out of memory errors**:
  - Reduced batch size from 128 → 64

- **Overfitting**:
  - Applied dropout, L2 regularization, and image augmentation
  - Used `EarlyStopping` and `ModelCheckpoint`

---

## ✅ Summary

We developed and compared CNN models for cancer detection in pathology slides. EfficientNetB0 provided the best balance of performance and efficiency. Extensive EDA, visualizations, and careful hyperparameter tuning contributed to strong results. Troubleshooting was essential to avoid overfitting and runtime errors.

---

## 📦 Requirements

- Python 3.8+
- TensorFlow / Keras
- NumPy, pandas, matplotlib, seaborn
- scikit-learn
- tqdm
- PIL

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🧪 Evaluation

- Validation Accuracy: ~90–95% depending on split
- AUC used to monitor performance
- No data leakage — separate validation split from training

---

## 📈 Results

- Trained CNN achieved solid results on validation
- Submission CSV generated with predictions for all 57,458 test images
- Format:
  ```csv
  id,label
  5e069ecb4ea52c12f80ce5a938dd1b3f,0
  924f4e52d60cfcbbfad6a7f601fa1f45,1
  ...


## 📂 GitHub Repository

You can explore the full project, notebook, and submission file in the GitHub repository:

**🔗 [GitHub Repository Link](https://github.com/abdallahmohammed2025/CNN_Histopathologic_Cancer_Detection)**  

The repository includes:

- 📓 `Histopathologic_Cancer_Detection_Full_Notebook.ipynb`: Complete training and inference notebook  
- 📁 `train/` and `test/`: Expected dataset structure  
- 📝 `submission.csv`: Ready-to-submit predictions  
- 📜 `README.md`: Full project documentation  
- 📌 `requirements.txt`

To clone the repository locally:

```bash
git clone https://github.com/abdallahmohammed2025/CNN_Histopathologic_Cancer_Detection.git