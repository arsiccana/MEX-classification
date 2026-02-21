---

# 🏋️ MEX Exercise Classification

**Author:** Ana Arsić
**Dataset:** MEx (Multimodal Exercise Dataset)
**Source:** [https://archive.ics.uci.edu/dataset/500/mex](https://archive.ics.uci.edu/dataset/500/mex)

---

## 📌 Project Overview

This project addresses a **multiclass classification problem** for recognizing physiotherapy exercises using the **MEx dataset**.

The goal is to classify the type of exercise (7 classes) based on sensor-derived numerical features.

The workflow includes:

* Data preprocessing and cleaning
* Feature scaling
* Dimensionality reduction
* Training multiple machine learning models
* Comparative performance analysis
* Model saving for reproducibility

---

## 📊 Feature Variants

Models were trained and evaluated on four different feature representations:

* **Full** – All extracted features (156 attributes)
* **PCA90** – Principal components preserving 90% variance
* **PCA95** – Principal components preserving 95% variance
* **KBest50** – Top 50 features selected using mutual information

---

## 🤖 Implemented Algorithms

The following classifiers were evaluated:

* Decision Tree
* Random Forest
* Logistic Regression
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Naive Bayes

---

## 📈 Evaluation Metrics

Models were evaluated using:

* **Accuracy**
* **F1-macro** (primary metric)
* **Confusion matrices** (train & test sets)

F1-macro was chosen because it treats all classes equally in the multiclass setting.

---

## 🏆 Best Model

The best overall performance was achieved using:

**Decision Tree (Full feature set)**

* Test Accuracy: 0.9922
* Test F1-macro: 0.9760

Random Forest also demonstrated strong robustness, especially on reduced feature sets.

---

## 📁 Repository Structure

### Data Files

* `mex_features_all_raw.csv` – Dataset after feature engineering (before preprocessing)
* `mex_features_all_preprocessed.csv` – Cleaned and imputed dataset


### Model Results

* `model_results_all.csv` – Results for all models and feature variants
* `best_per_variant.csv` – Best model for each feature representation

### Saved Models

* `best_model_overall__*.joblib`
* `best_model__*.joblib`

All models are saved in `.joblib` format.

---

## 🔁 Reproducibility

To reproduce results:

1. Clone the repository
2. Install required dependencies
3. Run `class_mex.ipynb`

### Required Libraries

* numpy
* pandas
* scikit-learn
* matplotlib
* joblib

---

## ⚙️ Loading a Saved Model

```python
import joblib

model = joblib.load("best_model_overall__...joblib")
predictions = model.predict(X)
```

---

## 📝 Notes

* Stratified train-test split (80:20) was used.
* StandardScaler was applied (fit on training data only).
* PCA and feature selection were performed after scaling.
* GridSearch was applied to optimize SVM hyperparameters.

---
