# Support Vector Machines (SVM)

## Table of Contents
1. [Introduction to SVM](#introduction-to-svm)
2. [Hard Margin SVM](#hard-margin-svm)
3. [Soft Margin SVM](#soft-margin-svm)
4. [Key Differences](#key-differences)
5. [Practical Considerations](#practical-considerations)

---

## Introduction to SVM

**Support Vector Machines (SVM)** is a powerful supervised learning algorithm used for both classification and regression tasks. The core idea of SVM is to find an optimal hyperplane that separates data points of different classes with the maximum possible margin.

### Key Concepts:
- **Hyperplane**: A decision boundary that separates the classes in the feature space
- **Support Vectors**: The data points that are closest to the hyperplane and define its position and orientation
- **Margin**: The distance between the hyperplane and the nearest data points (support vectors)

SVMs work effectively in both linear and non-linear settings through the use of kernel functions.

---

## Hard Margin SVM

### Overview
Hard Margin SVM assumes that the data is **linearly separable** - meaning we can perfectly separate the two classes without any misclassification. This approach works well when the data has clear separation between classes.

### Mathematical Formulation

#### Problem Statement
Given a dataset with $n$ samples: $\{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ where:
- $\mathbf{x}_i \in \mathbb{R}^d$ is a feature vector
- $y_i \in \{-1, +1\}$ is the class label

#### The Hyperplane
A hyperplane is defined as:
$$f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b = 0$$

where:
- $\mathbf{w} \in \mathbb{R}^d$ is the weight vector (normal to the hyperplane)
- $b \in \mathbb{R}$ is the bias term

#### Constraints (Linear Separability)
For perfect classification, every point must satisfy:
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 \quad \forall i = 1, 2, ..., n$$

This constraint ensures:
- Points of class +1 are on the positive side: $\mathbf{w}^T \mathbf{x}_i + b \geq 1$
- Points of class -1 are on the negative side: $\mathbf{w}^T \mathbf{x}_i + b \leq -1$

#### Margin Calculation
The margin (distance between parallel hyperplanes at the support vectors) is:
$$\text{Margin} = \frac{2}{\|\mathbf{w}\|}$$

#### Optimization Objective
To **maximize the margin**, we **minimize** $\|\mathbf{w}\|^2$:

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2$$

subject to:
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 \quad \forall i$$

#### Lagrange Dual Formulation
Using Lagrange multipliers $\alpha_i \geq 0$, the dual problem becomes:

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^{n} \alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j$$

subject to:
$$0 \leq \alpha_i \quad \text{and} \quad \sum_{i=1}^{n} \alpha_i y_i = 0$$

#### Solution
The optimal weight vector is:
$$\mathbf{w}^* = \sum_{i=1}^{n} \alpha_i^* y_i \mathbf{x}_i$$

The decision function is:
$$f(\mathbf{x}) = \text{sign}\left(\sum_{i=1}^{n} \alpha_i^* y_i \mathbf{x}_i^T \mathbf{x} + b^*\right)$$

### Advantages of Hard Margin
- ✅ Simple and efficient for linearly separable data
- ✅ Produces clean decision boundaries
- ✅ Interpretable solution

### Disadvantages of Hard Margin
- ❌ Fails if data is not perfectly linearly separable
- ❌ Sensitive to outliers
- ❌ Overfits when outliers or noise exist

---

## Soft Margin SVM

### Overview
Soft Margin SVM allows **some misclassification** to occur in training data. This makes the algorithm robust to outliers and noise, and applicable to real-world data that is rarely perfectly separable.

### Mathematical Formulation

#### Slack Variables
To allow violations of the hard margin constraint, we introduce **slack variables** $\xi_i \geq 0$ for each sample:

$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i \quad \forall i = 1, 2, ..., n$$

where:
- $\xi_i = 0$: Point is correctly classified and on the correct side of the margin
- $0 < \xi_i < 1$: Point is correctly classified but inside the margin
- $\xi_i \geq 1$: Point is misclassified on the wrong side of the hyperplane

#### Optimization Objective
The soft margin formulation adds a **penalty term** for violations:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \left(\frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n} \xi_i\right)$$

subject to:
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i \quad \forall i$$
$$\xi_i \geq 0 \quad \forall i$$

where:
- $C > 0$ is the **regularization parameter** (trade-off parameter):
  - Large $C$: Penalizes misclassifications heavily → stricter margin, potential overfitting
  - Small $C$: Allows more misclassifications → wider margin, potential underfitting

#### Lagrange Dual Formulation
Using Lagrange multipliers $\alpha_i$ and $\mu_i$:

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^{n} \alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j$$

subject to:
$$0 \leq \alpha_i \leq C \quad \forall i$$
$$\sum_{i=1}^{n} \alpha_i y_i = 0$$

#### Solution
The weight vector and decision function remain:
$$\mathbf{w}^* = \sum_{i=1}^{n} \alpha_i^* y_i \mathbf{x}_i$$

$$f(\mathbf{x}) = \text{sign}\left(\sum_{i=1}^{n} \alpha_i^* y_i \mathbf{x}_i^T \mathbf{x} + b^*\right)$$

#### Hinge Loss Interpretation
Soft margin SVM can also be viewed through the lens of **hinge loss**:

$$L(\mathbf{w}) = \sum_{i=1}^{n} \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b)) + \lambda\|\mathbf{w}\|^2$$

where:
- The max term is the hinge loss (zero if correct prediction with margin ≥ 1)
- $\lambda = \frac{1}{2C}$ is the regularization strength

### Advantages of Soft Margin
- ✅ Works with non-separable data
- ✅ Robust to outliers and noise
- ✅ Prevents overfitting through regularization
- ✅ Flexible through the $C$ parameter

### Disadvantages of Soft Margin
- ❌ Requires tuning the $C$ parameter
- ❌ May not achieve perfect separation if needed
- ❌ Slightly more complex than hard margin

---

## Key Differences

| Aspect | Hard Margin | Soft Margin |
|--------|------------|------------|
| **Data Assumption** | Linearly separable | Non-separable (realistic) |
| **Misclassification** | Not allowed ($\xi_i = 0$) | Allowed and penalized |
| **Constraints** | $y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1$ | $y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i$ |
| **Optimization** | $\min \frac{1}{2}\|\mathbf{w}\|^2$ | $\min \left(\frac{1}{2}\|\mathbf{w}\|^2 + C\sum \xi_i\right)$ |
| **Parameter Tuning** | No tuning required | Requires tuning $C$ |
| **Robustness** | Sensitive to outliers | Robust to outliers |
| **Practical Use** | Rarely (perfect separation is rare) | Most common (real-world data) |

---

## Practical Considerations

### 1. Kernel Trick
Both hard and soft margin SVMs can be extended to handle non-linear decision boundaries using the **kernel trick**:

$$\mathbf{x}_i^T \mathbf{x}_j \rightarrow K(\mathbf{x}_i, \mathbf{x}_j)$$

Common kernels:
- **Linear**: $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$
- **Polynomial**: $K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j + 1)^d$
- **RBF (Gaussian)**: $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2)$

### 2. Hyperparameter Selection

**For Soft Margin SVM:**
- Use **grid search** or **random search** to find optimal $C$
- Use **cross-validation** to avoid overfitting during tuning
- Typical range: $C \in [10^{-3}, 10^3]$

### 3. Feature Scaling
- Always **normalize/standardize** features (e.g., StandardScaler)
- SVM is sensitive to feature magnitudes
- Required for distance-based algorithms

### 4. Computational Complexity
- **Time Complexity**: $O(n^3)$ for hard margin, $O(n^3)$ for soft margin (using standard QP solvers)
- **Space Complexity**: $O(n^2)$ for storing kernel matrix
- For large datasets, consider approximate methods or use fast solvers

### 5. When to Use SVM
- ✅ Small to medium-sized datasets
- ✅ High-dimensional data
- ✅ Binary classification (naturally extends to multi-class)
- ✅ Clear margin separation needed
- ❌ Very large datasets (use SGD SVM or other alternatives)
- ❌ Highly unbalanced classes (use class weights)

---

## Summary

- **Hard Margin SVM** finds the maximum margin separator for linearly separable data with zero tolerance for misclassification
- **Soft Margin SVM** allows controlled misclassification through slack variables and a regularization parameter $C$, making it practical for real-world noisy data
- The mathematical frameworks are nearly identical, with soft margin being a generalization of hard margin
- Soft margin SVM is the industry standard due to its robustness and flexibility

