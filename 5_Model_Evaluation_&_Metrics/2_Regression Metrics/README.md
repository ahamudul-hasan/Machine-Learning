# Regression Metrics: MSE, MAE and RMSE

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

## 4. R² Score (Coefficient of Determination)

### What is R² Score?
R² Score measures how well your model explains the variation in the data. It tells you what percentage of the variance in the target variable is explained by your model. It ranges from 0 to 1 (or 0% to 100%).

### Mathematical Formula
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

Where:
- $SS_{res} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ = Sum of squared residuals (prediction errors)
- $SS_{tot} = \sum_{i=1}^{n} (y_i - \bar{y})^2$ = Total sum of squares (variance from mean)
- $\bar{y}$ = Mean of actual values

### Simple Explanation
Imagine predicting house prices:
- If R² = 0.85, it means your model explains 85% of the price variation
- The remaining 15% is unexplained (could be noise or missing features)
- An R² = 1.0 means perfect predictions
- An R² = 0.0 means your model is as good as just predicting the average price

Think of it as: **"What percentage of the ups and downs in prices does my model catch?"**

### Advantages ✅
- **Easy to interpret**: Percentage (0-100%) that everyone understands
- **Normalized metric**: Can compare across different datasets
- **Shows model quality**: Single number tells if model is good or bad
- **Considers baseline**: Compares against just predicting the mean

### Disadvantages ❌
- **Increases with more features**: Even useless features improve R²
- **Can be misleading**: R² = 0.9 doesn't mean perfect predictions
- **Sensitive to outliers**: One bad prediction drops R² significantly
- **Not appropriate for all cases**: Doesn't work well when y is not continuous

---

## 5. Adjusted R² Score

### What is Adjusted R²?
Adjusted R² is an improved version of R² that penalizes you for adding more features to the model. It prevents the problem of R² artificially improving when you add useless variables.

### Mathematical Formula
$$R_{adj}^2 = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

Where:
- $R^2$ = Original R² score
- $n$ = Number of data points
- $p$ = Number of features/predictors

### Simple Explanation
Imagine you're building a house price prediction model:
- Model 1: Uses 3 features (size, rooms, age) → R² = 0.85
- Model 2: Uses 10 features (above + color, number of windows, etc.) → R² = 0.86

Regular R² suggests Model 2 is better (0.86 > 0.85), but Adjusted R² might show:
- Model 1: Adjusted R² = 0.84
- Model 2: Adjusted R² = 0.83

Adjusted R² reveals that the extra features aren't helping!

Think of it as: **"Is the improvement worth adding more features?"**

### Advantages ✅
- **Prevents overfitting**: Penalizes unnecessary features
- **Fair comparison**: Compare models with different number of features
- **More realistic**: Shows true model quality, not inflated scores
- **Better for feature selection**: Helps choose which features to keep

### Disadvantages ❌
- **Harder to interpret**: Slightly more complex than R²
- **Can be negative**: If model is very bad, Adjusted R² can go below 0
- **Still has limitations**: Doesn't fix all R² problems
- **Not useful with few features**: Only matters when adding many features

---

## Quick Comparison: R² vs Adjusted R²

| Aspect | R² Score | Adjusted R² |
|--------|----------|-------------|
| **Range** | 0 to 1 | Can be negative to 1 |
| **Interpretation** | % of variance explained | Adjusted % (penalizes features) |
| **Adding Features** | Always increases or stays same | May decrease |
| **Use Case** | Quick model evaluation | Comparing models fairly |
| **Complexity** | Simple | Slightly complex |

---

## R² Score Interpretation Guide

| R² Value | Model Quality | Interpretation |
|----------|---------------|-----------------|
| **0.90 - 1.0** | Excellent | Model explains 90-100% of variance |
| **0.70 - 0.90** | Good | Model explains 70-90% of variance |
| **0.50 - 0.70** | Moderate | Model explains 50-70% of variance |
| **0.30 - 0.50** | Fair | Model explains 30-50% of variance |
| **0.0 - 0.30** | Poor | Model explains less than 30% of variance |
| **< 0** | Very Bad | Model worse than predicting the mean! |

---

## Example: R² and Adjusted R² Comparison

Suppose you have a dataset with 50 samples predicting house prices:

**Model 1** (3 features):
- $SS_{res} = 50,000$ (error)
- $SS_{tot} = 200,000$ (total variation)
- $R^2 = 1 - \frac{50,000}{200,000} = 0.75$
- $Adjusted R^2 = 1 - \frac{(1-0.75)(50-1)}{50-3-1} = 0.72$

**Model 2** (15 features):
- $SS_{res} = 40,000$ (slightly better error)
- $R^2 = 1 - \frac{40,000}{200,000} = 0.80$ ← Better!
- $Adjusted R^2 = 1 - \frac{(1-0.80)(50-1)}{50-15-1} = 0.70$ ← Worse!

**Conclusion**: Model 1 is actually better because it uses fewer features for similar performance!

---

## When to Use Each Metric

### Use **R² Score** when:
- You want a quick assessment of model fit
- Comparing models with the same number of features
- Presenting results to non-technical people
- Training simple models with few features

### Use **Adjusted R²** when:
- Comparing models with different number of features
- Trying to select the best features
- Building complex models and avoiding overfitting
- Want a fair comparison between different model architectures

### Use **MAE/RMSE + R² together** when:
- You want both error magnitude AND explanation quality
- Building production models (need both accuracy and interpretability)
- Presenting comprehensive model evaluation

---

## Key Takeaways

1. **MAE** = Average absolute errors (easiest to understand)
2. **MSE** = Average squared errors (best for optimization)
3. **RMSE** = Square root of MSE (best balance of both)
4. **R² Score** = What % of variance your model explains (0-100%)
5. **Adjusted R²** = R² with penalty for extra features

### Metric Selection Guide:
- **Error magnitude?** → Use **MAE** or **RMSE**
- **Model fit quality?** → Use **R²** or **Adjusted R²**
- **Training neural networks?** → Use **MSE** as loss function
- **Fair model comparison?** → Use **Adjusted R²**
- **Business reporting?** → Use **RMSE** + **R²** together
