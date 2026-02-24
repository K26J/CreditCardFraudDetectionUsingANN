# 🕵️ The Hunt for the Invisible: Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

> **"Finding fraud isn't hard. Finding fraud without accusing everyone else is the real challenge."**

## 📖 The Story Behind the Code

In a dataset of **284,807 credit card transactions**, only **492 were fraudulent**. That is just **0.17%**.

Most models would simply guess "Not Fraud" for every transaction and achieve **99.83% Accuracy**. While that looks great on paper, it's useless in production because it catches **zero** criminals.

My goal was to build a Neural Network that could find these "needles in the haystack" **without** relying on synthetic data generation (SMOTE). I wanted a model that learned real human behavior, not synthetic patterns.

---

## 📉 Chapter 1: The Data Reality (EDA)
Before writing a single line of modeling code, I had to understand the battlefield.
* **The Imbalance:** The classes were so skewed that standard training loops would simply ignore the fraud cases entirely.
* **The Features:** Most features (`V1`...`V28`) were already PCA-transformed and scaled, but `Amount` was not. I identified that `Amount` had a massive range that could confuse the neural network, so I applied `StandardScaler` to normalize it.
* **The "Tells":** Correlation analysis revealed that `V17`, `V14`, and `V12` were the strongest indicators of fraud.

---

## ⚔️ Chapter 2: The Struggle with Overfitting
I didn't get the right answer on the first try. The journey involved failing, analyzing, and pivoting.

### Attempt 1: The "Naive" Weights
I started by simply telling the model "Pay more attention to fraud." I calculated class weights mathematically.
* **Result:** The model caught some fraud (Recall 77%), but it panicked and flagged too many innocent people (Precision 84%). It wasn't aggressive enough.

### Attempt 2: The "Aggressive" Weights (1:10)
I manually forced the model to treat every fraud case as 10x more important than a normal transaction.
* **Result:** Recall shot up to **86%**, which was great!
* **The Problem:** The model started **overfitting**. It began memorizing the specific fraud examples in the training set rather than learning general rules. The validation loss started creeping up while training loss went down.

---

## 🚀 Chapter 3: The Solution (Architecture Engineering)
To fix the overfitting *without* losing the high detection rate, I redesigned the architecture. I didn't need *more* data (SMOTE); I needed a *smarter* model.

I introduced two key defenses:
1.  **L2 Regularization:** Punished the model for creating overly complex decision boundaries.
2.  **Dropout Layers:** Randomly disabled neurons during training to force the network to be robust and independent.

### The Final Architecture
```python
model_f1 = Sequential()

# Input Layer + Hidden Layer 1
model_f1.add(Dense(64, activation='relu', input_dim=29))

# Hidden Layer 2: The "Filter"
# Added L2 Regularization to stop weights from exploding
model_f1.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model_f1.add(Dropout(0.2)) # 20% of neurons are dropped to prevent memorization

# Hidden Layer 3: The "Refiner"
model_f1.add(Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model_f1.add(Dropout(0.5)) # High dropout to force generalization

# Output
model_f1.add(Dense(1, activation='sigmoid'))