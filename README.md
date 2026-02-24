# 🕵️ The Hunt for the Invisible: Credit Card Fraud Detection (ANN)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

> **"Finding fraud isn't hard. Finding fraud without accusing everyone else is the real challenge."**

## 📖 The Story Behind the Project
In a dataset of **284,807 credit card transactions**, only **492 were fraudulent**. That is just **0.17%**.

Most standard models would simply guess "Not Fraud" for every transaction and achieve **99.83% Accuracy**. While that looks great on paper, it is useless in production because it catches **zero** criminals.

My goal was to build a Neural Network that could find these "needles in the haystack" **without** relying on synthetic data generation (SMOTE). I wanted a model that learned real human behavior, not synthetic patterns.

---

## 📊 Chapter 1: The Data Reality (EDA)
**Source:** Kaggle Credit Card Fraud Detection Dataset  
**Dimensions:** 284,807 rows × 31 columns  

### 🔍 Key Discovery: The "Imbalance Trap"
* **Class 0 (Non-Fraud):** 284,315 entries
* **Class 1 (Fraud):** 492 entries
* **Constraint:** The classes are so skewed that standard training loops ignore the fraud cases entirely.

### 📉 Feature Engineering
1.  **Scaling:** Most features (`V1`...`V28`) were already PCA-transformed and scaled. However, the `Amount` feature was not.
    * *Action:* Applied `StandardScaler` to `Amount` to prevent it from dominating the gradients.
2.  **The "Tells":** Correlation analysis revealed that `V17`, `V14`, `V12`, `V10`, and `V16` were the strongest indicators of fraud.

---

## ⚔️ Chapter 2: The Modeling Journey (Evolution of Experiments)
I didn't get the right answer on the first try. The journey involved failing, analyzing, and pivoting.

| Iteration | Strategy | Result | Verdict |
| :--- | :--- | :--- | :--- |
| **Base Model** | Simple ANN (10 Epochs) | **Recall: 80% | Precision: 92%** | Good start, but lacked generalization. |
| **Model 2** | Calculated Class Weights | **Recall: 77% | Precision: 84%** | **Failed.** Pure math-based weights didn't capture the nuance. |
| **Model 3** | Aggressive Weights (1:10) | **Recall: 86% | Precision: 67%** | **Unstable.** Great detection, but too many false alarms and overfitting. |
| **Model 4** | Weights (1:10) + Tuning | **Recall: 85% | Precision: 72%** | **Better**, but overfitting persisted on validation data. |
| **Final Model** | **Weights (1:10) + L2 Regularization + Dropout** | **Recall: 85% | Precision: 78%** | **SUCCESS.** Stable loss curves, no overfitting. |

---

## 🚀 Chapter 3: The Solution (Architecture Engineering)
To fix the overfitting observed in Models 3 & 4 *without* losing the high detection rate, I redesigned the architecture. I didn't need *more* data (SMOTE); I needed a *smarter* model.

### The Final Architecture
```python
model_f1 = Sequential()

# Input Layer + Hidden Layer 1
model_f1.add(Dense(64, activation='relu', input_dim=29))

# Hidden Layer 2: The "Filter"
# Added L2 Regularization (0.001) to stop weights from exploding
model_f1.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model_f1.add(Dropout(0.2)) # 20% of neurons are dropped to prevent memorization

# Hidden Layer 3: The "Refiner"
model_f1.add(Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model_f1.add(Dropout(0.5)) # High dropout to force generalization

# Output
model_f1.add(Dense(1, activation='sigmoid'))