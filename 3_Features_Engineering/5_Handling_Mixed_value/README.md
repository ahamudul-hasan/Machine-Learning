# 🔧 Feature Transformation Toolkit
---

## 📋 Overview

This module provides hands-on examples of **data preprocessing and feature transformation** techniques for machine learning pipelines. Each notebook demonstrates core concepts and best practices using scikit-learn's transformer ecosystem.

---


## 🎯 Box-Cox vs Yeo-Johnson: Deep Dive

Both are **power transforms** that reshape numeric features to approximate a normal distribution, improving model performance especially for linear algorithms.

### Box-Cox Power Transform

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='box-cox')
X_transformed = pt.fit_transform(X + 1e-6)  # Add small constant for safety
```

**Characteristics:**
- ✅ Highly effective for positive-skewed data
- ❌ **Requires strictly positive values** $(x > 0)$
- Searches for optimal $\lambda$ parameter
- Common in financial and time-series data

**When to use:**
- All features are guaranteed to be positive (prices, ages, durations)
- You want maximum normalization power

---

### Yeo-Johnson Power Transform

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')  # Default method
X_transformed = pt.fit_transform(X)
```

**Characteristics:**
- ✅ Handles zero and negative values gracefully
- ✅ More robust for mixed-sign data
- ✅ **Recommended as default choice**
- Slightly less powerful normalization than Box-Cox

**When to use:**
- Features may contain zeros or negative values
- You want a "safe default" without preprocessing checks
- Working with centered/standardized data

---

### Core Concepts Covered

| Concept | Description |
|---------|-------------|
| **FunctionTransformer** | Apply custom Python functions as sklearn transformers |
| **ColumnTransformer** | Apply different transformations to different feature columns |
| **Pipeline** | Chain preprocessing steps and models to prevent data leakage |
| **PowerTransformer** | Normalize features using Box-Cox or Yeo-Johnson methods |

---

## 📚 Notebook Guide

### 1️⃣ `functional_transformer.ipynb`
**Focus:** Custom transformations with FunctionTransformer

- **Dataset:** Titanic survival data (`Age`, `Fare`, `Survived`)
- **Key Topics:**
  - Applying custom functions (`np.log1p`) to normalize skewed features
  - Visualizing distributions before/after transformation
  - Column-specific transformations with `ColumnTransformer`
  
**Use Case:** When you need to apply mathematical transformations (log, sqrt, etc.) to reduce skewness in features like pricing or time data.

---

### 2️⃣ `Column_transformer.ipynb`
**Focus:** Efficient handling of mixed feature types

- **Dataset:** COVID toy dataset (mixed numeric & categorical)
- **Approach:**
  - ❌ **Manual method:** separate imputation + encoding + concatenation
  - ✅ **Best practice:** unified `ColumnTransformer`

**Learning:** Why automated column transformers prevent errors and improve code readability.

---

### 3️⃣ `Pipelines/Without_Pipeline.ipynb`
**Focus:** Common preprocessing pitfalls

- **Dataset:** Titanic survival data
- **Demonstrates:**
  - Manual preprocessing steps without a pipeline structure
  - Risk of data leakage if not careful with train/test splits
  - Difficulty in reproducing exact transformations

⚠️ **Warning:** Shows the "wrong way" to highlight why pipelines matter.

---

### 4️⃣ `Pipelines/with_pipeline.ipynb`
**Focus:** Production-grade preprocessing workflows

- **Dataset:** Titanic survival data
- **Pipeline Components:**
  - Data imputation
  - One-hot encoding
  - Feature scaling
  - Feature selection (`SelectKBest`)
  - Model training (Decision Tree)

**Best Practice:** `Pipeline(...)` vs `make_pipeline(...)` syntax comparison.

---

### 5️⃣ `Power_Transformer.ipynb`
**Focus:** Power transformations for regression tasks

- **Dataset:** Concrete strength prediction
- **Transformations:**
  - **Box-Cox Transform:** For strictly positive values
  - **Yeo-Johnson Transform:** For mixed/negative values
- **Output:** λ (lambda) parameters for each feature, performance comparisons

---


## 📊 Quick Comparison Table

| Aspect | Box-Cox | Yeo-Johnson |
|--------|---------|------------|
| **Positive values only** | ✅ Required | ✅ Optional |
| **Handles zeros** | ❌ No | ✅ Yes |
| **Handles negatives** | ❌ No | ✅ Yes |
| **Normalization power** | 🔥 Strong | 💪 Good |
| **Default choice** | ❌ No | ✅ Yes |

---

## 🚀 Quick Start

### Setup
```python
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline

# Load data
X = pd.read_csv('data.csv')

# Option 1: Box-Cox (for positive values)
transformer = PowerTransformer(method='box-cox')
X_transformed = transformer.fit_transform(X + 1e-6)

# Option 2: Yeo-Johnson (for mixed values)
transformer = PowerTransformer(method='yeo-johnson')
X_transformed = transformer.fit_transform(X)
```

### In a Pipeline (Recommended)
```python
from sklearn.linear_model import LinearRegression

pipe = Pipeline([
    ('transform', PowerTransformer(method='yeo-johnson')),
    ('model', LinearRegression())
])

pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```

---

## ⚡ Best Practices

| ✅ Do | ❌ Don't |
|------|---------|
| Use transformers inside pipelines | Fit transformers on combined train+test data |
| Apply `fit_transform` only on training data | Refit transformers on test data |
| Use Yeo-Johnson as default | Assume Box-Cox works with all data types |
| Check feature distributions | Skip visualization of transformations |
| Version your transformer objects | Share unfitted transformer instances |

---

## 📁 Project Structure

```
3_Transformer/
├── README.md                                  # This file
├── functional_transformer.ipynb              # Log transforms & ColumnTransformer
├── Column_transformer.ipynb                  # Manual vs automated approaches
├── Power_Transformer.ipynb                   # Box-Cox & Yeo-Johnson comparison
└── Pipelines/
    ├── Without_Pipeline.ipynb                # ⚠️ Anti-pattern demonstration
    └── with_pipeline.ipynb                   # ✅ Best practice example
```

---

## 🔗 Data Sources

| Notebook | Dataset | Shape | Purpose |
|----------|---------|-------|---------|
| functional_transformer | `train.csv` (Titanic) | 891 × 2 | Survival prediction |
| Column_transformer | `covid_toy.csv` | - | Mixed features handling |
| Pipelines | `train.csv` (Titanic) | 891 × 11 | Classification pipeline |
| Power_Transformer | `concrete_data.csv` | 1,030 × 8 | Strength regression |

All datasets are located in `../../../Data/` directory.

---

## 💡 Common Issues & Solutions

### `FileNotFoundError` when loading data
```python
# Check your notebook location, then adjust path accordingly
# From: 3_Transformer/ → Data/ requires: ../../../Data/
```

### Box-Cox fails with zero/negative values
```python
# Solution 1: Use Yeo-Johnson
pt = PowerTransformer(method='yeo-johnson')

# Solution 2: Add small constant to Box-Cox
pt = PowerTransformer(method='box-cox')
X_transformed = pt.fit_transform(X + 1e-6)
```

### Model performance doesn't improve after transformation
- ✓ Not all data benefits from transformation (especially tree-based models)
- ✓ Consider feature scaling instead for distance-based models
- ✓ Check if transformations are applied correctly to test data

---

## 🎓 Key Takeaways

1. **Always use Pipelines** to prevent data leakage and ensure reproducibility
2. **Understand your data** — Box-Cox vs Yeo-Johnson depends on value ranges
3. **Visualize transformations** — Use histograms and Q-Q plots to verify effectiveness
4. **Test thoroughly** — Not all models benefit from the same transformations
5. **Document applied parameters** — Save λ values and transformer objects for deployment

---

<div align="center">

**Happy Transforming! 🎉**

</div>