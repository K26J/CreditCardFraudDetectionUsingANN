# Credit Card Fraud Detection using Artificial Neural Networks (ANN)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

## 📌 Project Overview
This project implements a Deep Learning model to detect fraudulent credit card transactions. The primary challenge was the **severe class imbalance** (0.17% fraud cases). 

Unlike traditional approaches that rely on synthetic data augmentation (SMOTE), this project focuses on a **data-centric approach using Class Weights and Regularization**. The final model achieves a stable trade-off between Recall (capturing fraud) and Precision (minimizing false alarms) on purely real-world data.

---

## 📊 Dataset & EDA
**Source:** Kaggle Credit Card Fraud Detection Dataset  
**Dimensions:** 284,807 rows × 31 columns  

### Key Insights
* **Imbalance:** The dataset is highly skewed.
    * `Class 0` (Non-Fraud): 284,315
    * `Class 1` (Fraud): 492
* **Features:**
    * `V1` to `V28`: Principal Component Analysis (PCA) transformed features (Scaled).
    * `Amount`: Required scaling during preprocessing.
* **Correlation:** EDA revealed that `V17`, `V14`, `V12`, `V10`, and `V16` are the strongest predictors of fraud.

---

## 🛠️ The Modeling Journey (Iterative Process)
*The following table documents the experiments conducted to arrive at the final robust model.*

| Iteration | Approach | Observations & Results | Verdict |
| :--- | :--- | :--- | :--- |
| **Baseline** | Basic ANN (10 Epochs) | High Precision (92%), Good Recall (80%). | **Good start**, but needed better generalization. |
| **Model 2** | Calculated Class Weights | Precision dropped to 84%, Recall 77%. | **Underwhelming.** Pure math-based weights didn't capture the nuance. |
| **Model 3** | Manual Weights (1:10) | Recall increased to 86%, Precision dropped to 67%. | **Unstable.** The model began to overfit significantly. |
| **Model 4** | Weights (1:10) + Tuning | Recall 85%, Precision 72%. | **Better**, but overfitting persisted on validation data. |
| **Final Model** | **Weights (1:10) + L2 Regularization + Dropout** | **Recall: 85% | Precision: 78%** | **SUCCESS.** Stable loss curves, no overfitting. |

---

## 🧠 Final Model Architecture
To combat the overfitting observed in Models 3 and 4, the final architecture incorporates **L2 Regularization** and **Dropout layers**.

```python
model_f1 = Sequential()

# Input Layer + Hidden Layer 1
model_f1.add(Dense(64, activation='relu', input_dim=29))

# Hidden Layer 2 with L2 Regularization
model_f1.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model_f1.add(Dropout(0.2)) # Prevent neuron co-adaptation

# Hidden Layer 3
model_f1.add(Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model_f1.add(Dropout(0.5))

# Output Layer
model_f1.add(Dense(1, activation='sigmoid'))

# Compiler
model_f1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Precision', 'Recall'])