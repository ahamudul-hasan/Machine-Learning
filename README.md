<div align="center">

# 🤖 Machine Learning Roadmap

> A structured, end-to-end journey through Machine Learning — from data fundamentals to production deployment.

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.4.2-013243?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-3.0.1-150458?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.8-11557c?logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-4c72b0?logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=Jupyter&logoColor=white)
![VS Code](https://img.shields.io/badge/VS%20Code-007ACC?logo=visualstudiocode&logoColor=white)

</div>

---

## 🗺️ Table of Contents

| # | Section | Key Topics |
|:-:|---------|------------|
| 1 | [📥 Data Gathering](#3--data-gathering) | CSV · JSON · API · Web Scraping |
| 2 | [🔍 Data Analysis](#4--data-analysis) | Understanding Data · EDA |
| 3 | [⚙️ Features Engineering](#5-%EF%B8%8F-features-engineering) | Scaling · Encoding · Imputation · Outliers |
| 4 | [🤖 Models](#6--models) | Supervised · Unsupervised |
| 5 | [📏 Model Evaluation & Metrics](#7--model-evaluation--metrics) | Classification · Regression · Cross-Validation |
| 6 | [🔧 Hyperparameter Tuning](#8--hyperparameter-tuning) | Grid Search · Random · Bayesian |
| 7 | [🧠 Neural Networks & Deep Learning](#9--neural-networks--deep-learning) | ANN · CNN · RNN · Frameworks |
| 8 | [🚀 Model Deployment](#10--model-deployment) | APIs · Docker · Cloud · MLOps |

---

## 1. 📥 Data Gathering

| # | Source | Method |
|:-:|--------|--------|
| 1 | 📄 CSV | Local / Online file reading |
| 2 | 🔷 JSON | JSON parsing |
| 3 | 🔌 API | REST API fetching |
| 4 | 🌐 Web | Web scraping |

---

## 2. 🔍 Data Analysis

| # | Topic | Description |
|:-:|-------|-------------|
| 1 | Understanding Data | Data types, shape, descriptive stats, null values |
| 2 | Univariate EDA | Distribution of individual features |
| 3 | Multivariate EDA | Relationships between multiple features |

---

## 3. ⚙️ Features Engineering

<details>
<summary><strong>3.1 — Feature Transformation</strong></summary>
<br>

**📏 Scaling**

| # | Technique |
|:-:|-----------|
| 1 | Standardization |
| 2 | Normalization |

**🏷️ Handling Categorical Features**

| # | Technique |
|:-:|-----------|
| 1 | Ordinal Encoding |
| 2 | One Hot Encoding |

**🔩 Pipelines**

| # | Technique | Variant |
|:-:|-----------|---------|
| 1 | Column Transformer | Without |
| 2 | Column Transformer | Using |
| 3 | Pipeline | Without |
| 4 | Pipeline | Using |

**🔁 Transformers**

| # | Technique | Variant |
|:-:|-----------|---------|
| 1 | Functional Transformer | Without |
| 2 | Functional Transformer | Using |
| 3 | Power Transformer | Without |
| 4 | Power Transformer | Using |

**🔢 Handling Numerical Features**

| # | Technique | Variant |
|:-:|-----------|---------|
| 1 | Binning | Without |
| 2 | Binning | Using |
| 3 | Binarization | — |

</details>

<details>
<summary><strong>3.2 — Missing Data Imputation</strong></summary>
<br>

**Univariate — Numerical**

| # | Technique |
|:-:|-----------|
| 1 | Mean-Median Imputation |
| 2 | Arbitrary Imputation |
| 3 | Random Sample Imputation |

**Univariate — Categorical**

| # | Technique |
|:-:|-----------|
| 1 | Frequent Value Imputation |
| 2 | Missing Column |
| 3 | Random Sample Imputation |

**Multivariate**

| # | Technique |
|:-:|-----------|
| 1 | KNN Imputation |
| 2 | Iterative Imputation |

</details>

<details>
<summary><strong>3.3 — Outlier Detection</strong></summary>
<br>

| # | Technique | Variant |
|:-:|-----------|---------|
| 1 | Z-Score Filtering | Mean Standardization |
| 2 | Z-Score Filtering | Z-Scoring |
| 3 | IQR Filtering | — |
| 4 | Percentile Filtering | — |

</details>

<details>
<summary><strong>3.4 — Other Transformations</strong></summary>
<br>

| # | Topic |
|:-:|-------|
| 1 | Handling Mixed Values |
| 2 | Handling Date & Time |
| 3 | Complete Case Analysis |
| 4 | Feature Construction & Splitting |
| 5 | Feature Importance (Random Forest) |

</details>

---

## 4. 🤖 Models

### 4.1 Supervised Learning

<details>
<summary><strong>📐 Linear Regression</strong></summary>
<br>

| # | Algorithm | Implementation |
|:-:|-----------|:-------------:|
| 1 | Simple Linear Regression (1D) | Built-in |
| 2 | Simple Linear Regression (1D) | Own Class |
| 3 | Multiple Linear Regression | Built-in |
| 4 | Multiple Linear Regression | Own Class |
| 5 | Polynomial Regression | Built-in |
| 6 | Polynomial Regression | Own Class |

</details>

<details>
<summary><strong>📉 Gradient Descent</strong></summary>
<br>

| # | Algorithm | Implementation |
|:-:|-----------|:-------------:|
| 1 | Gradient Descent — Single Column | Built-in |
| 2 | Gradient Descent — Single Column | Own Class |
| 3 | Batch Gradient Descent | Built-in |
| 4 | Batch Gradient Descent | Own Class |
| 5 | Stochastic Gradient Descent | Built-in |
| 6 | Stochastic Gradient Descent | Own Class |
| 7 | Mini-Batch Gradient Descent | Built-in |
| 8 | Mini-Batch Gradient Descent | Own Class |

</details>

<details>
<summary><strong>🧷 Regularization</strong></summary>
<br>

| # | Algorithm | Implementation |
|:-:|-----------|:-------------:|
| 1 | Ridge Regression | Built-in |
| 2 | Ridge Regression | Own Class (1D) |
| 3 | Ridge Regression | Own Class (ND) |
| 4 | Lasso Regression | Built-in |
| 5 | ElasticNet Regression | Built-in |

</details>


<details>
<summary><strong>🔀 Logistic Regression</strong></summary>
<br>

| # | Topic | Variant |
|:-:|-------|---------|
| 1 | Perceptron Trick | Step Function |
| 2 | Perceptron Trick | Sigmoid Function |
| 3 | Binary Classification | Gradient Descent |
| 4 | Accuracy Metrics | Binary Class |
| 5 | Accuracy Metrics | Multi-class |
| 6 | Multinomial Classification | Softmax |
| 7 | Non-linear Logistic Regression | Polynomial Features |

</details>

<details>
<summary><strong>🌳 Decision Tree</strong></summary>
<br>

| # | Type |
|:-:|------|
| 1 | Classification |
| 2 | Regression |

</details>

<details>
<summary><strong>📊 Naive Bayes</strong></summary>
<br>

| # | Topic |
|:-:|-------|
| 1 | Implementation |

</details>

<details>
<summary><strong>📍 K-Nearest Neighbours (KNN)</strong></summary>
<br>

| # | Type |
|:-:|------|
| 1 | Classification |

</details>

<details>
<summary><strong>⚡ Support Vector Machine (SVM)</strong></summary>
<br>

| # | Type |
|:-:|------|
| 1 | Classification |

</details>

<details>
<summary><strong>🌲 Ensemble Learning</strong></summary>
<br>

**Voting**

| # | Type |
|:-:|------|
| 1 | Voting Classifier |
| 2 | Voting Regressor |

**Bagging**

| # | Technique |
|:-:|-----------|
| 1 | Bootstrapping — With Replacement |
| 2 | Bootstrapping — Pasting |
| 3 | Bootstrapping — Random Subspace |
| 4 | Bootstrapping — Random Patch | — |
| 5 | Bagging Classification |
| 6 | Bagging Regression |
| 7 | Random Forest — Bootstrapping |
| 8 | Random Forest — Plotting |
| 9 | Random Forest — SearchCV |

**Boosting**

| # | Algorithm | Topic |
|:-:|-----------|-------|
| 1 | AdaBoost | Implementation |
| 2 | AdaBoost | Hyperparameter Tuning |
| 3 | Gradient Boosting | Implementation |
| 4 | Gradient Boosting | Additive Modelling — Regression |
| 5 | Gradient Boosting | Additive Modelling — Classification |
| 6 | XGBoost | Implementation |

**Stacking**

| # | Technique |
|:-:|-----------|
| 1 | Blending Stacking |

</details>

### 4.2 Unsupervised Learning

| # | Algorithm |
|:-:|-----------|
| 1 | K-Means Clustering |
| 2 | Hierarchical Clustering |
| 3 | Dimensionality Reduction (PCA) |

---

## 5. 📏 Model Evaluation & Metrics

<details>
<summary><strong>5.1 Classification Metrics</strong></summary>
<br>

| Metric | Description |
|--------|-------------|
| Accuracy | Fraction of correct predictions |
| Precision | TP / (TP + FP) — exactness of positive predictions |
| Recall (Sensitivity) | TP / (TP + FN) — coverage of actual positives |
| F1-Score | Harmonic mean of Precision & Recall |
| Confusion Matrix | Table of TP, TN, FP, FN |
| ROC Curve | True Positive Rate vs False Positive Rate |
| AUC Score | Area under the ROC curve (1.0 = perfect) |
| Log Loss | Penalizes confident wrong predictions |
| Classification Report | Full summary of all class-wise metrics |

</details>

<details>
<summary><strong>5.2 Regression Metrics</strong></summary>
<br>

| Metric | Formula |
|--------|---------|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² Score | Proportion of variance explained by the model |
| Adjusted R² | R² adjusted for number of features |
| MAPE | Mean Absolute Percentage Error |

</details>

<details>
<summary><strong>5.3 Cross-Validation</strong></summary>
<br>

| Method | Description |
|--------|-------------|
| K-Fold | Split into K folds, train/test K times |
| Stratified K-Fold | Preserves class distribution across folds |
| Leave-One-Out (LOO) | Extreme case: 1 sample as test each time |
| Repeated K-Fold | Repeat K-Fold multiple times with different splits |
| `cross_val_score` | scikit-learn utility for CV scoring |

</details>

<details>
<summary><strong>5.4 Bias-Variance Tradeoff & Learning Curves</strong></summary>
<br>

| Concept | Description |
|---------|-------------|
| Underfitting | High bias, low variance — model too simple |
| Overfitting | Low bias, high variance — model memorises training data |
| Sweet Spot | Balance via regularization, more data, or correct complexity |
| Learning Curves | Training vs Validation error over dataset size |

</details>

---

## 6. 🔧 Hyperparameter Tuning

| # | Method | Description |
|:-:|--------|-------------|
| 1 | Grid Search CV | Exhaustive search over all parameter combinations |
| 2 | Randomized Search CV | Random sampling of parameter space — faster |
| 3 | Bayesian Optimization | Probabilistic model-guided search (e.g., `optuna`) |

**Key Concepts:** `param_grid` · `param_distributions` · `cv` folds · `scoring` · `best_params_` · `best_estimator_` · `refit`

---