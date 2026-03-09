<div align="center">

# 🤖 Machine Learning

> A structured, end-to-end journey through Machine Learning — from Python fundamentals to advanced models.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.4.2-013243?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-3.0.1-150458?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.8-11557c?logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-4c72b0?logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?logo=scikit-learn&logoColor=white)

</div>

---

## 📚 Table of Contents

| # | Topic |
|---|-------|
| 1 | [Python Libraries](#1-python-libraries) |
| 2 | [Data Gathering](#2-data-gathering) |
| 3 | [Data Analysis](#3-data-analysis) |
| 4 | [Features Engineering](#4-features-engineering) |
| 5 | [Models](#5-models) |
| 6 | [Model Evaluation & Metrics](#6-model-evaluation--metrics) |
| 7 | [Hyperparameter Tuning](#7-hyperparameter-tuning) |
| 8 | [Neural Networks & Deep Learning](#8-neural-networks--deep-learning) |
| 9 | [Model Deployment](#9-model-deployment) |

---

## 1. Python Libraries

### 📐 NumPy

<details>
<summary><strong>Click to expand</strong></summary>

| Topic | Details |
|---|---|
| Arrays | 1D, 2D, nD array creation |
| Array Operations | Indexing, slicing, reshaping, stacking |
| Math Operations | Add, subtract, multiply, dot product, broadcasting |
| Statistical Functions | mean, median, std, var, min, max |
| Linear Algebra | `np.linalg` — inverse, determinant, eigenvalues |
| Random Module | `np.random` — random arrays, seeds, distributions |
| Boolean Indexing | Filtering arrays with conditions |

</details>

### 🐼 Pandas

<details>
<summary><strong>Click to expand</strong></summary>

| Topic | Details |
|---|---|
| Series & DataFrame | Creating, reading, inspecting |
| Data I/O | CSV, JSON, Excel, SQL |
| Indexing & Selection | `loc`, `iloc`, boolean filtering |
| Handling Missing Data | `isnull`, `dropna`, `fillna` |
| Data Manipulation | `merge`, `concat`, `groupby`, `pivot_table` |
| Apply & Lambda | Row/column-wise transformations |
| String Operations | `str` accessor methods |
| DateTime | Parsing dates, `dt` accessor, resampling |
| Aggregation | `agg`, `describe`, `value_counts` |

</details>

### 📊 Matplotlib

<details>
<summary><strong>Click to expand</strong></summary>

| Analysis Type | Charts |
|---|---|
| Univariate | Histogram |
| Univariate & Bivariate | Pie Chart |
| Bivariate & Multivariate | 2D Line Plot, 3D Line Plot, Scatter Plot, 3D Scatter Plot, 3D Surface Plot, Contour Plots, Heatmap, Bar Chart |
| Layout | Subplot |

</details>

### 🎨 Seaborn

<details>
<summary><strong>Click to expand</strong></summary>

**Relation Plots**
- Scatter Plot · Facet Plot · Line Plot

**Distribution Plots**
- Histogram · Facet Histogram · Bivariate Histogram
- KDE Plot · Bivariate KDE Plot · Rug Plot

**Categorical Plots**

| Category | Plots |
|---|---|
| Scatter (Bivariate) | Stripplot, Swarmplot |
| Distribution (Univariate) | Boxplot, Violinplot |
| Estimate (Central Tendency) | Barplot, Pointplot, Countplot |

**Matrix Plots**
- Heatmap · Clustermap

**Regression Plot**

**Multi-Plots**
- Facet Plot · Joint Plot · Pair Plot

</details>

### 📈 Plotly

<details>
<summary><strong>Click to expand</strong></summary>

| Chart Type | Use Case |
|---|---|
| Scatter Plot | Relationship between variables |
| Line Chart | Trends over time |
| Bar Chart | Category comparisons |
| Histogram | Distribution of values |
| Box Plot | Spread and outliers |
| Heatmap | Correlation / matrix data |
| 3D Scatter / Surface | 3-dimensional data |
| Pie / Donut Chart | Part-to-whole proportions |
| Animated Charts | Time-series animation with `animation_frame` |
| Subplots | `make_subplots` for multi-panel dashboards |

</details>

---

## 2. Data Gathering

| Source | Method |
|--------|--------|
| 📄 CSV | Local / Online file reading |
| 🔷 JSON | JSON parsing |
| 🔌 API | REST API fetching |
| 🌐 Web | Web scraping |

---

## 3. Data Analysis

### 🔍 Understanding Data
- Data types, shape, descriptive statistics, null values

### 📊 Exploratory Data Analysis (EDA)

| Type | Description |
|------|-------------|
| Univariate | Distribution of individual features |
| Multivariate | Relationships between multiple features |

---

## 4. Features Engineering

### 🔧 4.1 Feature Transformation

<details>
<summary><strong>Feature Scaling</strong></summary>

- Standardization
- Normalization

</details>

<details>
<summary><strong>Handling Categorical Features</strong></summary>

- Ordinal Encoding
- One Hot Encoding

</details>

<details>
<summary><strong>Transformers</strong></summary>

- Column Transformer
- Functional Transformer
- Power Transformer
- **Pipelines**
  - Without Pipeline
  - Using Pipeline

</details>

<details>
<summary><strong>Handling Numerical Features</strong></summary>

- Binning
- Binarization

</details>

<details>
<summary><strong>Handling Mixed Values & DateTime</strong></summary>

- Handling Mixed Values
- Handling Date & Time
- Complete Case Analysis

</details>

<details>
<summary><strong>Missing Data Imputation</strong></summary>

**Univariate**
- *Numerical:* Mean-Median Imputation, Arbitrary Imputation
- *Categorical:* Frequent Value Imputation, Missing Column
- *Other:* Random Sample Imputation, Missing Indicator, Automatic Imputation

**Multivariate**
- KNN Multivariate Imputation
- Iterative Imputation

</details>

<details>
<summary><strong>Outlier Detection</strong></summary>

- Z-Score Filtering
- IQR Filtering
- Percentile Filtering

</details>

### 🏗️ 4.2 Feature Construction & Splitting
### ⭐ 4.3 Feature Importance
- Feature Importance using Random Forest

### 🔻 4.4 PCA (Principal Component Analysis)

---

## 5. Models

### 🧭 5.1 Supervised Learning

<details>
<summary><strong>Gradient Descent</strong></summary>

| Variant | Implementation |
|---------|---------------|
| Gradient Descent | Built-in · Own Class |
| Batch Gradient Descent | — |
| Stochastic Gradient Descent | Built-in · Own Class |
| Mini-Batch Gradient Descent | Built-in · Own Class |

</details>

<details>
<summary><strong>Linear Regression</strong></summary>

| Type | Implementation |
|------|---------------|
| Simple Linear Regression | Built-in · Own Class |
| Regression Metrics | — |
| Multiple Linear Regression | Built-in · Own Class |
| Polynomial Regression | Built-in · Own Class |
| Ridge Regression | Built-in · Own Class · Own Class (Multidimensional) |
| Lasso Regression | — |

</details>

<details>
<summary><strong>Logistic Regression</strong></summary>

- **Perceptron Trick** — Step Function · Sigmoid Function
- **Gradient Descent Based**
  - Binary Classification
  - Accuracy Score Metrics (Binary & Multi-class)
  - Multinomial / Multiclass Classification (Softmax)
  - Polynomial / Non-linear Logistic Regression

</details>

<details>
<summary><strong>Decision Tree</strong></summary>

- Classification
- Regression

</details>

<details>
<summary><strong>Naive Bayes</strong></summary>
</details>

<details>
<summary><strong>K-Nearest Neighbours (KNN)</strong></summary>

- Classification

</details>

<details>
<summary><strong>Support Vector Machine (SVM)</strong></summary>

- Classification

</details>

<details>
<summary><strong>Ensemble Learning</strong></summary>

**Voting**
- Voting Classifier · Voting Regressor

**Bagging**
- Bootstrapping: With Replacement · Pasting · Random Subspace · Random Patch
- Bagging Classification · Bagging Regression
- **Random Forest**
  - Bootstrapping in Random Forest
  - Algorithm · Plotting · SearchCV

**Boosting**
| Algorithm | Details |
|-----------|---------|
| AdaBoost | Implementation, Hyperparameter Tuning |
| Gradient Boosting | Regression & Classification Additive Modelling |
| XGBoost | — |

**Stacking**
- Blending Stacking

</details>

---

### 🔵 5.2 Unsupervised Learning

| Algorithm | Type |
|-----------|------|
| K-Means Clustering | Partitioning |
| Hierarchical Clustering | Agglomerative |

---

## 6. Model Evaluation & Metrics

### 📏 6.1 Classification Metrics

<details>
<summary><strong>Click to expand</strong></summary>

| Metric | Description |
|--------|-------------|
| Accuracy | Fraction of correct predictions |
| Precision | TP / (TP + FP) — how exact positive predictions are |
| Recall (Sensitivity) | TP / (TP + FN) — how many positives were caught |
| F1-Score | Harmonic mean of Precision & Recall |
| Confusion Matrix | Table of TP, TN, FP, FN |
| ROC Curve | True Positive Rate vs False Positive Rate |
| AUC Score | Area under the ROC curve (closer to 1 = better) |
| Log Loss | Penalizes confident wrong predictions |
| Classification Report | Full summary of all class-wise metrics |

</details>

### 📐 6.2 Regression Metrics

<details>
<summary><strong>Click to expand</strong></summary>

| Metric | Formula / Description |
|--------|-----------------------|
| MAE (Mean Absolute Error) | Average absolute difference |
| MSE (Mean Squared Error) | Average squared difference |
| RMSE (Root MSE) | Square root of MSE |
| R² Score | Proportion of variance explained by the model |
| Adjusted R² | R² adjusted for number of features |
| MAPE | Mean Absolute Percentage Error |

</details>

### 🔁 6.3 Cross-Validation

<details>
<summary><strong>Click to expand</strong></summary>

- **K-Fold Cross-Validation** — split data into K folds, train/test K times
- **Stratified K-Fold** — preserves class distribution across folds
- **Leave-One-Out (LOO)** — extreme case: 1 sample as test each time
- **Repeated K-Fold** — repeat K-Fold multiple times with different splits
- **`cross_val_score`** — scikit-learn utility for CV scoring

</details>

### ⚖️ 6.4 Bias-Variance Tradeoff
- **Underfitting** — high bias, low variance (model too simple)
- **Overfitting** — low bias, high variance (model memorises training data)
- **Sweet Spot** — balance via regularization, more data, or correct model complexity

### 📊 6.5 Learning Curves
- Training vs Validation error over dataset size
- Diagnosing underfitting and overfitting visually

---

## 7. Hyperparameter Tuning

### 🔍 7.1 Search Strategies

| Method | Description |
|--------|-------------|
| **Grid Search CV** | Exhaustive search over all parameter combinations |
| **Randomized Search CV** | Random sampling of parameter space — faster |
| **Bayesian Optimization** | Probabilistic model-guided search (e.g., `optuna`) |

### ⚙️ 7.2 Key Concepts
- `param_grid` and `param_distributions`
- `cv` parameter — number of cross-validation folds
- `scoring` — metric used to evaluate each combination
- `best_params_` and `best_estimator_`
- `refit` — automatically refit on full data with best params

### 🧪 7.3 Practical Usage
- Using `GridSearchCV` with pipelines
- Using `RandomizedSearchCV` for large search spaces
- Early stopping in gradient boosting (XGBoost, LightGBM)

---

## 8. Neural Networks & Deep Learning

### 🧠 8.1 Fundamentals

<details>
<summary><strong>Click to expand</strong></summary>

| Concept | Description |
|---------|-------------|
| Perceptron | Single neuron — linear decision boundary |
| Activation Functions | Sigmoid, ReLU, Tanh, Softmax, Leaky ReLU |
| Forward Propagation | Input → hidden layers → output |
| Backpropagation | Computing gradients and updating weights |
| Loss Functions | Binary Cross-Entropy, Categorical Cross-Entropy, MSE |
| Optimizers | SGD, Adam, RMSProp, AdaGrad |
| Learning Rate | Controls step size during gradient descent |
| Epochs & Batch Size | Training iterations and data chunk size |

</details>

### 🏗️ 8.2 Neural Network Architecture

<details>
<summary><strong>Click to expand</strong></summary>

- **Artificial Neural Network (ANN)** — fully connected layers
  - Binary Classification
  - Multi-class Classification
  - Regression
- **Convolutional Neural Network (CNN)** — spatial feature extraction
  - Conv2D, MaxPooling, Flatten
  - Image Classification
- **Recurrent Neural Network (RNN)** — sequential data
  - Vanilla RNN
  - LSTM (Long Short-Term Memory)
  - GRU (Gated Recurrent Unit)
  - Time Series Forecasting
- **Autoencoders** — unsupervised representation learning

</details>

### 🛠️ 8.3 Regularization & Optimization
- Dropout
- Batch Normalization
- Weight Initialization (Xavier, He)
- Early Stopping
- L1 / L2 Regularization in neural networks

### 📦 8.4 Frameworks
| Framework | Use Case |
|-----------|----------|
| **TensorFlow / Keras** | High-level DL, production-ready |
| **PyTorch** | Research-friendly, dynamic computation graph |

---

## 9. Model Deployment

### 💾 9.1 Saving & Loading Models

| Method | Library | Use Case |
|--------|---------|----------|
| Pickle | `pickle` | General Python object serialization |
| Joblib | `joblib` | Optimized for NumPy arrays (sklearn models) |
| ONNX | `onnx` | Cross-framework model format |
| SavedModel | TensorFlow | Production TF model format |

### 🌐 9.2 Serving Models as APIs

<details>
<summary><strong>Click to expand</strong></summary>

- **Flask** — lightweight Python web framework
  - Creating a `/predict` endpoint
  - JSON request / response handling
- **FastAPI** — modern async API framework
  - Pydantic data validation
  - Auto-generated Swagger docs
- **Streamlit** — rapid ML demo dashboards
  - Input widgets, charts, model output display

</details>

### 🐳 9.3 Containerization & Cloud
- **Docker** — containerize your ML application
  - `Dockerfile`, `requirements.txt`, `docker build`
- **Cloud Deployment**
  - AWS (SageMaker, Lambda + API Gateway)
  - Google Cloud (Vertex AI, Cloud Run)
  - Azure (Azure ML, Azure Functions)
- **CI/CD for ML** — automated testing and redeployment pipelines

### 📊 9.4 Monitoring & MLOps
| Concept | Tool / Practice |
|---------|----------------|
| Experiment Tracking | MLflow, Weights & Biases |
| Data Versioning | DVC (Data Version Control) |
| Model Registry | MLflow Model Registry |
| Feature Store | Feast, Tecton |
| Drift Detection | Monitor input/output distribution shifts |

---
