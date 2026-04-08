# Lasso Regression (L1 Regularization)

## Table of Contents
1. [Theoretical Concept](#theoretical-concept)
2. [Why Use Lasso Regression](#why-use-lasso-regression)
3. [Mathematical Explanation](#mathematical-explanation)
4. [4 Key Practical Points](#4-key-practical-points)
5. [Limitations](#limitations)
6. [When to Use Lasso](#when-to-use-lasso)

---

## Theoretical Concept

### Simple Explanation
Lasso Regression is a technique that helps prevent machine learning models from overfitting by **shrinking some model weights to zero**. It's like adding a penalty for having too many features or having very large coefficients in your model.

**Regular Regression** tries to fit the data as closely as possible.

**Lasso Regression** tries to fit the data while also keeping the model simple by reducing the number of features.

### How It Works
Lasso adds a "cost" for having large coefficients:
- If a feature is not important, Lasso pushes its coefficient towards 0
- If a coefficient reaches exactly 0, that feature is effectively removed from the model
- This is called **feature selection** - Lasso automatically decides which features to keep

### The Name
**LASSO** stands for **Least Absolute Shrinkage and Selection Operator**
- **Least**: Minimize prediction error
- **Absolute**: Use absolute values (not squared)
- **Shrinkage**: Reduce coefficient sizes
- **Selection**: Remove unimportant features

---

## Why Use Lasso Regression

### 1. **Prevents Overfitting**
   - Regular models can memorize training data noise
   - Lasso forces simplicity by penalizing large coefficients
   - Better performance on new, unseen data

### 2. **Feature Selection**
   - Automatically identifies important features
   - Removes irrelevant features (coefficient = 0)
   - Reduces model complexity and interpretability problems

### 3. **Better Interpretability**
   - Fewer features = easier to understand what the model is doing
   - Can focus on the most important predictors

### 4. **Handles High-Dimensional Data**
   - When you have many features (e.g., 1000 features)
   - Lasso reduces this naturally through feature elimination
   - Much faster predictions with fewer features

### 5. **Improves Generalization**
   - Models perform better on new data
   - Avoids fitting to noise in training data

---

## Mathematical Explanation

### The Cost Function

#### Standard Linear Regression (Ordinary Least Squares)
$$J(\beta) = \frac{1}{2m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)})^2$$

Where:
- $m$ = number of training samples
- $h(x^{(i)})$ = predicted value
- $y^{(i)}$ = actual value

**Problem**: Can fit noise and overfit

#### Lasso Regression (L1 Regularization)
$$J(\beta) = \frac{1}{2m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} |w_j|$$

Where:
- First term: **Prediction error** (same as regular regression)
- Second term: **Penalty for large coefficients** (L1 regularization)
- $\lambda$ = regularization strength (hyperparameter)
- $|w_j|$ = absolute value of each coefficient
- $n$ = number of features

### Understanding the Terms

#### Prediction Error Term
$$\frac{1}{2m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)})^2$$
- Measures how well the model fits the data
- Wants this to be small (good predictions)

#### Penalty Term
$$\lambda \sum_{j=1}^{n} |w_j|$$
- Penalizes large coefficients
- Encourages coefficients to be small or zero
- The $\lambda$ controls how much penalty to apply

### The Role of Lambda ($\lambda$)

| $\lambda$ Value | Effect |
|---|---|
| $\lambda = 0$ | No penalty → Overfitting (regular regression) |
| $\lambda = 0.001$ | Weak penalty → Some shrinkage, less feature selection |
| $\lambda = 0.1$ | Moderate penalty → Good balance |
| $\lambda = 1$ | Strong penalty → Many features removed (underfitting) |
| $\lambda = 100$ | Very strong penalty → Simple model, but loses information |

### Why Absolute Value (|w|) Instead of Squared (w²)?

**Lasso uses absolute value:**
$$\lambda \sum_{j=1}^{n} |w_j|$$

**Ridge uses squared values:**
$$\lambda \sum_{j=1}^{n} w_j^2$$

**Key Difference:**
- **L1 (Lasso)**: Can push coefficients to exactly 0 → Feature elimination
- **L2 (Ridge)**: Shrinks coefficients toward 0 but rarely reaches exactly 0 → All features retained

**Visual Example:**
```
Regular Regression:  w1 = 5,   w2 = 3,   w3 = 1
Ridge Regression:    w1 = 4.8, w2 = 2.9, w3 = 0.9
Lasso Regression:    w1 = 3.5, w2 = 0,   w3 = 0    (features 2 & 3 removed!)
```

### The Optimization Problem

Lasso solves:
$$\text{minimize } \frac{1}{2m} ||y - Xw||^2 + \lambda ||w||_1$$

Subject to:
$$\sum_{j=1}^{n} |w_j| \leq t$$

Where $||w||_1 = \sum_{j=1}^{n} |w_j|$ is the L1 norm

**Interpretation:** Find coefficients that minimize error while keeping the sum of absolute values of coefficients below a threshold.

---

## 4 Key Practical Points

### 1. How are coefficients affected?
As regularization strength ($\lambda$ or `alpha`) increases, Lasso shrinks coefficients toward zero. For small values of `alpha`, most coefficients remain non-zero but reduced in magnitude. For larger values, many coefficients become exactly zero.

In practice:
- `alpha = 0` behaves like ordinary linear regression (no L1 penalty)
- Small `alpha` values reduce overfitting while keeping most predictors
- Large `alpha` values produce sparse models by removing weak predictors

### 2. Higher coefficients are affected more
Features with larger absolute coefficients typically experience stronger shrinkage as `alpha` increases. While all coefficients are penalized by the L1 term, less important features usually hit zero first, and strong features survive longer.

In practice:
- Coefficient paths show progressive shrinkage as `alpha` grows
- Weak predictors are eliminated early
- Dominant predictors remain non-zero until stronger regularization is applied

### 3. Impact on bias and variance
Increasing regularization generally increases bias and decreases variance. This is the core bias-variance trade-off:
- Low `alpha`: low bias, high variance (risk of overfitting)
- Moderate `alpha`: balanced bias and variance (often best generalization)
- High `alpha`: high bias, low variance (risk of underfitting)

The best `alpha` is usually where expected prediction error is minimized, not where bias or variance alone is minimized.

### 4. Effect of regularization on loss function
Lasso adds an L1 penalty to the ordinary loss:

$$
J(w) = \text{MSE}(w) + \lambda \sum_{j=1}^{n} |w_j|
$$

As $\lambda$ increases, the optimizer prefers smaller coefficients, shifting the minimum of the total objective toward simpler models. This is why the model becomes more stable and sparse, but can also underfit if regularization is too strong.

In practice:
- The objective curve changes with each `alpha`
- Larger penalties flatten/shift the optimum toward smaller coefficient values
- Excessive regularization can hurt predictive performance

---

## Limitations

### 1. **Feature Selection is Unstable**
   - With highly correlated features, Lasso randomly selects one and ignores others
   - If you remove/add correlated features, results change significantly
   - **Solution**: Use Elastic Net (combines L1 and L2)

### 2. **Hyperparameter Tuning Required**
   - Must choose the right $\lambda$ value
   - Wrong $\lambda$ causes underfitting or overfitting
   - Requires cross-validation to find optimal value

### 3. **Not Ideal for Very High Dimensions**
   - Slower computation with extremely many features
   - May select features arbitrarily among correlated features

### 4. **Loses Information**
   - By forcing coefficients to zero, you lose information about removed features
   - May discard features that could be useful in combination

### 5. **Interpretation Challenges**
   - Zero coefficients don't necessarily mean features are unimportant
   - May be confounded by correlated features

### 6. **Convergence Warnings**
   - Can have convergence issues with high-degree polynomial features
   - **Solution**: Scale features with StandardScaler

---

## When to Use Lasso

### ✅ Use Lasso When:
- You have **many features** (high-dimensional data)
- You believe **only some features are important**
- You want **automatic feature selection**
- Interpretability is important
- You have **correlated features** and want to select subset
- Dataset is **prone to overfitting**

### ❌ Don't Use Lasso When:
- You have **few features** (not much benefit)
- All features are **truly important**
- You have **many highly correlated features** (use Elastic Net instead)
- You need **exact feature inclusion** (hard to control which features are selected)

---

## Comparison: Lasso vs Ridge vs Regular Regression

| Aspect | Regular Regression | Ridge (L2) | Lasso (L1) |
|--------|-------------------|-----------|-----------|
| **Penalty** | None | Sum of squares | Sum of absolute values |
| **Feature Selection** | No | No | Yes |
| **Zero Coefficients** | No | Rare | Yes |
| **Interpretability** | Good | Good | Excellent |
| **Correlated Features** | Keeps all | Shrinks all | Selects one randomly |
| **Computation Speed** | Fast | Fast | Medium |
| **Overfitting Risk** | High | Low | Low |

---

## Key Takeaways

1. **Lasso shrinks some coefficients to exactly zero** → automatic feature selection
2. **Lambda controls the strength of regularization** → balance between fit and simplicity
3. **Uses absolute values (L1)** → enables feature elimination (unlike Ridge's L2)
4. **Best with high-dimensional data** → reduces complexity naturally
5. **Requires hyperparameter tuning** → use cross-validation to find best lambda
6. **Watch for correlated features** → Lasso picks one arbitrarily

---

## Example

```
Dataset: House prices with 50 features (square feet, rooms, location, etc.)

Regular Regression:
- Uses all 50 features
- May overfit to training data noise
- Hard to understand which features matter

Lasso Regression (λ=0.1):
- Automatically removes 30 unimportant features (coefficients = 0)
- Uses only 20 most important features
- Better prediction on new houses
- Easy to interpret: "These 20 features drive house prices"
```

