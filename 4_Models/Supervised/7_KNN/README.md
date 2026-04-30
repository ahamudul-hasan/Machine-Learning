# K-Nearest Neighbors (KNN)

## Table of Contents
1. [Overview](#overview)
2. [How KNN Works](#how-knn-works)
3. [Selecting the K Value](#selecting-the-k-value)
4. [Decision Surface](#decision-surface)
5. [Overfitting and Underfitting](#overfitting-and-underfitting)
6. [Limitations of KNN](#limitations-of-knn)

---

## Overview

**K-Nearest Neighbors (KNN)** is a simple, non-parametric, lazy learning algorithm used for both classification and regression tasks. It works on the principle that similar data points are located close to each other in the feature space.

### Key Characteristics:
- **Non-parametric**: Makes no assumptions about the underlying data distribution
- **Lazy Learning**: No training phase; computation happens at prediction time
- **Instance-based**: Stores entire training dataset for predictions
- **Simple to understand and implement**: Intuitive approach based on similarity

---

## How KNN Works

### Algorithm Steps:

1. **Calculate Distance**: Compute the distance between the query point and all training points
   - Common distance metrics: Euclidean, Manhattan, Minkowski

2. **Find K Nearest**: Identify the K closest training points to the query point

3. **Aggregate**: 
   - **Classification**: Use majority voting (most common class label)
   - **Regression**: Use average of the K nearest values

4. **Predict**: Return the aggregated result

### Example:
```
For K=3: If nearest 3 neighbors are [Class A, Class A, Class B] 
→ Prediction = Class A (majority)
```

---

## Selecting the K Value

Choosing K is crucial for model performance. Different K values affect the bias-variance tradeoff.

### Guidelines for Selecting K:

| K Value | Effect | Best For |
|---------|--------|----------|
| **K = 1** | Uses only nearest point; very flexible | Capturing local patterns |
| **K = 3-5** | Small K; considers few neighbors | Small datasets, local patterns |
| **K = 10-20** | Moderate K; balanced approach | Most practical scenarios |
| **Large K** | Uses many neighbors; smoother boundary | Reducing noise, stable predictions |

### Selection Methods:

#### 1. **Odd K for Binary Classification**
- Avoids ties when voting
- Example: Use K=3, 5, 7 instead of K=2, 4, 6

#### 2. **Elbow Method (Grid Search)**
```python
scores = []
for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

# Plot and find the "elbow" point
plt.plot(range(1, 31), scores)
plt.xlabel('K value')
plt.ylabel('Accuracy')
plt.show()
```

#### 3. **Cross-Validation**
```python
from sklearn.model_selection import cross_val_score

best_k = None
best_score = 0

for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5)
    mean_score = scores.mean()
    
    if mean_score > best_score:
        best_score = mean_score
        best_k = k
```

#### 4. **Square Root Rule**
```
K = sqrt(n)  where n = number of training samples
```

### Impact of K:
- **K too small**: Model is too flexible, prone to overfitting
- **K too large**: Model becomes too rigid, prone to underfitting
- **K = n (total samples)**: Predicts class of majority regardless of input

---

## Decision Surface

The decision surface is the boundary that separates different classes in the feature space.

### Decision Surface Characteristics:

#### Small K (K=1)
```
Decision Boundary: Highly irregular and complex
- Creates small regions around each training point
- Follows the training data very closely
- Lots of zigzag patterns and curves
```

#### Large K (K=n)
```
Decision Boundary: Smooth and generalized
- Creates large regions for each class
- Ignores local details and noise
- Simple, clean boundaries
```

### Visualization Example:
```
K=1 (Complex):              K=5 (Moderate):          K=20 (Smooth):
████ ░ ████                 █████ ░ ░░░░░            █████ ░░░░░░░
██ ░░ ░ ██                  ████████░░░░░            ███████░░░░░░
█ ░░░░░░ █                  ████░░░░░░░░░            ███░░░░░░░░░░
█ ░░░░░░ █                  ███░░░░░░░░░░            ███░░░░░░░░░░
██ ░░ ░ ██                  ████░░░░░░░░░            ███░░░░░░░░░░
████ ░ ████                 █████░░░░░░░░            █████░░░░░░░░

(█ = Class A, ░ = Class B)
```

### Real Example Code:
```python
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

# Create synthetic 2D data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0)

# Train models with different K values
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, k in enumerate([1, 5, 20]):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    
    # Create mesh for decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[idx].contourf(xx, yy, Z, alpha=0.6)
    axes[idx].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    axes[idx].set_title(f'K={k}')
```

---

## Overfitting and Underfitting

### Overfitting in KNN (K too small)

**Definition**: Model learns training data too well, including noise and specific examples.

**Characteristics:**
- High accuracy on training data
- Low accuracy on test data
- Decision boundary is very complex and jagged
- Reacts to individual training points

**Example (K=1):**
```
- Accuracy on Training Set: 100% (memorizes all data)
- Accuracy on Test Set: 85% (fails on new data)
- Problem: Noise and outliers significantly affect predictions
```

**When it happens:**
- K is very small (K=1, 2)
- Training set is very small
- Feature space is high-dimensional

### Underfitting in KNN (K too large)

**Definition**: Model is too simple and fails to capture the underlying pattern.

**Characteristics:**
- Lower accuracy on both training and test data
- Decision boundary is overly smooth
- Misses important local patterns
- All predictions cluster around majority class

**Example (K=n):**
```
- Accuracy on Training Set: 75% (ignores data details)
- Accuracy on Test Set: 74% (too rigid)
- Problem: Cannot distinguish between classes
```

**When it happens:**
- K is very large (close to n)
- Training set is very noisy
- Classes are well-separated but K is too large

### Bias-Variance Tradeoff:

```
                    Model Complexity
     Low K                          High K
     ↓                              ↓
     
Error  ╱╲                        ╱──────
       │ ╲  Total Error         │    ╲─╲
       │  ╲                    ╱  ──╱   ╲╲
       │   ╲    ╱──╲         ╱ ╱─      ╱ ╲
       │    ╲  ╱    ╲─────  ╱─╱      ╱    
       │     ╲╱            ╱        ╱     
       │                          
       └──────────────────────────
       
Overfitting Zone        Optimal Zone      Underfitting Zone
(High Variance)      (K = 3 to 10)      (High Bias)
```

### Detecting Overfitting/Underfitting:

```python
from sklearn.model_selection import cross_val_score

train_errors = []
test_errors = []

for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Training error
    train_pred = knn.fit(X_train, y_train).predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    train_errors.append(1 - train_acc)
    
    # Test error
    test_pred = knn.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_errors.append(1 - test_acc)

plt.plot(range(1, 31), train_errors, label='Train Error')
plt.plot(range(1, 31), test_errors, label='Test Error')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.legend()
plt.show()

# If train error is much lower than test error → Overfitting
# If both errors are high → Underfitting
```

---

## Limitations of KNN

### 1. **Computational Cost at Prediction Time**
- **Problem**: Must calculate distance to ALL training points for each prediction
- **Complexity**: O(n × m) where n = training samples, m = features
- **Impact**: Slow on large datasets
- **Solution**: Use KD-trees, Ball trees, or approximate nearest neighbors

```python
# Standard KNN: Slow
knn = KNeighborsClassifier(n_neighbors=5)

# Optimized with KD-tree: Faster
knn = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
```

### 2. **Memory Requirements**
- **Problem**: Stores entire training dataset
- **Impact**: High memory usage with millions of samples
- **Limitation**: Cannot use for massive datasets efficiently

```
Training Data Size: 1 million samples × 100 features
Memory: ~400 MB (just for data storage)
```

### 3. **Curse of Dimensionality**
- **Problem**: Performance deteriorates in high-dimensional spaces
- **Why**: All points become equally distant as dimensions increase
- **Impact**: K=5 in 2D is different from K=5 in 50D

```
Distance in 2D:     Points clustered and different
Distance in 100D:   Almost all points equally far
```

**Solution**: Feature selection/reduction
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X)
```

### 4. **Sensitivity to Feature Scaling**
- **Problem**: Features with large ranges dominate distance calculation
- **Example**: Age (0-100) vs Income (10,000-100,000)

```
Without Scaling:
Distance = sqrt((Age_diff)² + (Income_diff)²)
           = sqrt(1 + 900000) ≈ 950  (income dominates)

With Scaling:
Both normalized to 0-1 range
Distance = sqrt((Age_norm)² + (Income_norm)²)  (equal contribution)
```

**Solution**: Always scale features
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 5. **Imbalanced Datasets**
- **Problem**: Majority class dominates voting
- **Example**: If 95% samples are Class A, predictions tend towards Class A

```
Dataset: 950 samples Class A, 50 samples Class B
K=5 in a dense Class A region:
5 nearest neighbors = [A, A, A, A, A] → Always predicts A
```

**Solution**: Weight neighbors by distance
```python
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
# Closer neighbors have more influence
```

### 6. **Sensitive to Outliers**
- **Problem**: Outliers can be nearest neighbors and distort predictions
- **Impact**: One outlier can cause misclassification

```
Normal Data: Class A, Class A, Class A, Class A
Outlier: Class B (misplaced point)

K=5: If outlier is one of 5 nearest → Wrong prediction
```

**Solution**: Remove outliers or use weighted KNN
```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()  # Less sensitive to outliers
X_scaled = scaler.fit_transform(X)
```

### 7. **No Clear Feature Importance**
- **Problem**: Cannot determine which features matter most
- **Impact**: Black box model - harder to interpret
- **Limitation**: Difficult for feature selection or business insights

### 8. **Poor Performance on Sparse Data**
- **Problem**: Works poorly with sparse vectors (many zeros)
- **Why**: Distance calculations are unreliable with sparse data
- **Example**: Text classification with TF-IDF vectors

### 9. **Ties in Voting**
- **Problem**: With even K, multiple classes can tie
- **Example**: K=4, votes = [A, A, B, B] (tie between A and B)

**Solution**: Use odd K
```python
knn = KNeighborsClassifier(n_neighbors=5)  # Odd number
```

### 10. **Not Suitable for Regression with Few Samples**
- **Problem**: Averaging K values can be unstable with small datasets
- **Impact**: High variance in predictions

---

## Summary Table

| Aspect | Detail |
|--------|--------|
| **Best Use** | Small to medium datasets, non-linear boundaries |
| **Worst Use** | Large datasets, high dimensions, sparse data |
| **Time Complexity** | Training: O(n), Prediction: O(n × m) |
| **Space Complexity** | O(n × m) |
| **Ideal K Range** | 3-20 (depends on dataset size) |
| **Parameter Tuning** | Only K; also consider distance metric, weights |
| **Feature Scaling** | Essential |
| **Handles Imbalance** | With weighted KNN |
| **Interpretability** | Low (black box) |

---

