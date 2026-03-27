# Regression Metrics: MSE, MAE, and RMSE

This guide explains the three most important metrics used to evaluate regression models: Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

---

## 1. MAE (Mean Absolute Error)

### What is MAE?
MAE measures the average distance between predicted values and actual values. It tells you how far off your predictions are, on average.

### Mathematical Formula
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

Where:
- $n$ = number of predictions
- $y_i$ = actual value
- $\hat{y}_i$ = predicted value
- $|...|$ = absolute value (ignore negative signs)

### Simple Explanation
Imagine you're predicting house prices:
- Actual price: $300,000
- Predicted price: $250,000
- Error: |300,000 - 250,000| = $50,000

If you have 10 predictions and the average error is $50,000, your MAE = $50,000.

### Advantages ✅
- **Easy to understand**: Error in same units as the target variable (e.g., dollars, meters)
- **Fair to all errors**: Small and large errors treated equally
- **Resistant to outliers**: Large errors don't have extra influence

### Disadvantages ❌
- **Less sensitive to large errors**: Doesn't penalize big mistakes heavily
- **Not differentiable**: Difficult to use in some optimization algorithms
- **Harder to backpropagate**: Problematic when training neural networks

---

## 2. MSE (Mean Squared Error)

### What is MSE?
MSE squares each error before averaging. Squaring makes larger errors much more noticeable.

### Mathematical Formula
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Where:
- $n$ = number of predictions
- $y_i$ = actual value
- $\hat{y}_i$ = predicted value

### Simple Explanation
Using the same house price example:
- Actual: $300,000
- Predicted: $250,000
- Error: $(300,000 - 250,000)^2 = 2,500,000,000$

Notice how squaring makes the error MUCH larger. This is intentional!

### Advantages ✅
- **Penalizes large errors**: One big mistake gets heavily punished
- **Smooth function**: Easy to use in optimization algorithms
- **Good for neural networks**: Works well with backpropagation

### Disadvantages ❌
- **Hard to interpret**: Units are squared (e.g., dollars² for house prices) making it confusing
- **Sensitive to outliers**: One very wrong prediction can inflate the metric
- **Biased towards large errors**: May not be fair if you care equally about all errors

---

## 3. RMSE (Root Mean Squared Error)

### What is RMSE?
RMSE is MSE with a square root applied. This converts MSE back to original units, making it interpretable like MAE.

### Mathematical Formula
$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

Or simply: $RMSE = \sqrt{MSE}$

### Simple Explanation
Continuing the house price example:
- MSE = 2,500,000,000
- RMSE = √2,500,000,000 = $50,000

Now it's back in dollars and easy to understand!

### Advantages ✅
- **Easy to interpret**: Same units as the target variable
- **Penalizes large errors**: Still punishes big mistakes harder than MAE
- **Popular metric**: Widely used in machine learning competitions and research
- **Good balance**: Between interpretability and error sensitivity

### Disadvantages ❌
- **Still sensitive to outliers**: Square root doesn't fully fix the outlier problem like MAE
- **Harder to explain than MAE**: Square root step can be confusing for non-technical people
- **Slightly harder to optimize**: Not quite as smooth as MSE for optimization

---

## Quick Comparison

| Metric | Formula | Units | Outlier Sensitivity | Interpretability |
|--------|---------|-------|-------------------|------------------|
| **MAE** | $\frac{1}{n}\sum \|error\|$ | Same as target ✅ | Low ✅ | Easy ✅ |
| **MSE** | $\frac{1}{n}\sum error^2$ | Squared ❌ | High ❌ | Hard ❌ |
| **RMSE** | $\sqrt{\frac{1}{n}\sum error^2}$ | Same as target ✅ | Medium ⚠️ | Easy ✅ |

---

## When to Use Each Metric

### Use **MAE** when:
- Domain outliers should NOT be heavily penalized
- You need simple, business-friendly explanation
- Errors should be treated fairly regardless of size
- Example: Predicting customer arrival times (5 min error = same impact everywhere)

### Use **MSE** when:
- You want to heavily penalize large errors
- Using it as a loss function for optimization
- You're working with neural networks
- Example: Predicting critical measurements (1° error is worse than expected)

### Use **RMSE** when:
- You want MSE's properties but need interpretability
- The metric needs to be in original units
- It's the standard metric in your field (very common!)
- You want a balanced approach between MAE and MSE

---

## Example Calculation

Suppose we have 3 actual and predicted values:

| Actual | Predicted | Error | Error² | \|Error\| |
|--------|-----------|-------|--------|-----------|
| 10 | 8 | -2 | 4 | 2 |
| 20 | 22 | 2 | 4 | 2 |
| 30 | 28 | -2 | 4 | 2 |

**MAE** = (2 + 2 + 2) / 3 = **2**

**MSE** = (4 + 4 + 4) / 3 = **4**

**RMSE** = √4 = **2**

---

## Key Takeaways

1. **MAE** = Average absolute errors (easiest to understand)
2. **MSE** = Average squared errors (best for optimization)
3. **RMSE** = Square root of MSE (best balance of both)

Choose based on your needs:
- Need fairness? → Use **MAE**
- Training a model? → Use **MSE** as loss function
- Reporting results? → Use **RMSE** (most common)
