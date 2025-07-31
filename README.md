# 🧬 Histopathologic Cancer Detection

This project addresses the **Kaggle competition** [Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection), where the objective is to detect metastatic cancer in small patches of tissue images taken from lymph node sections.

Using a **Convolutional Neural Network (CNN)** and transfer learning via **EfficientNetB0**, we train a binary classifier to identify cancerous tissue based on labeled histopathology images.

---

## 📁 Dataset

- **Source**: Provided by Kaggle
- **Image size**: 96x96 RGB patches
- **Classes**:  
  - `0`: No tumor  
  - `1`: Tumor present

- **Files**:
  - `train_labels.csv`: Contains image IDs and binary labels
  - `train/`: Folder containing ~220,000 image tiles
  - `test/`: Folder with ~57,458 unlabeled image tiles to predict

---

## 🧠 Model Architecture

We used **EfficientNetB0** with pre-trained `imagenet` weights:

- Input size: `96x96x3`
- Base model: `EfficientNetB0 (include_top=False)`
- Head:
  - `GlobalAveragePooling2D`
  - `Dropout(0.3)`
  - `Dense(1, activation='sigmoid')`

---

## 🛠️ Training Pipeline

- Framework: TensorFlow / Keras
- Data Augmentation: Horizontal/Vertical flips, rotations, brightness
- Optimizer: Adam
- Loss: Binary Crossentropy
- Metrics: Accuracy, AUC
- Callbacks: `EarlyStopping`, `ReduceLROnPlateau`
- Epochs: 10 (can be adjusted)
- Batch Size: 64

---

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