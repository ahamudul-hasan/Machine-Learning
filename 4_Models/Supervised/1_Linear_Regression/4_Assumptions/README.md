# Linear Regression Assumptions

Linear regression has several key assumptions that must be checked to ensure the model is reliable and valid. Here are the 5 main assumptions:

---

## 1. **Linear Relationship**

The relationship between the independent variables (features) and the dependent variable (target) should be **linear**.

### In simple terms:
- When you plot the features against the target, the data points should roughly form a straight line pattern.
- If the relationship is curved or scattered randomly, linear regression may not be the best model.

### Why does it matter?
- Linear regression assumes a straight-line relationship. If the actual relationship is curved, the model will perform poorly.
- You need to check this before building the model to ensure you're using the right approach.

### How to check:
- Create scatter plots of each feature vs. the target variable.
- Look for a linear (straight) pattern in the plots.

---

## 2. **Multicollinearity**

**Multicollinearity** occurs when two or more independent variables (features) are highly correlated with each other.

### In simple terms:
- Imagine having two features that measure almost the same thing (e.g., height in cm and height in inches).
- These features provide redundant information to the model.

### Why does it matter?
- When features are too similar, it becomes hard for the model to determine which feature actually influences the target.
- This can make the coefficients unstable and unreliable.
- The model may perform well on training data but poorly on new data.

### How to check:
- Calculate the **Variance Inflation Factor (VIF)** for each feature. VIF > 5-10 indicates multicollinearity.
- Create a correlation heatmap to see if features are highly correlated with each other.

### How to fix:
- Remove one of the correlated features.
- Combine correlated features into a single feature.
- Use dimensionality reduction techniques like PCA.

---

## 3. **Normality of Residuals**

The **residuals** (errors/differences between actual and predicted values) should follow a **normal distribution** (bell curve).

### In simple terms:
- After making predictions, calculate the error for each prediction (actual - predicted).
- These errors should be distributed like a bell curve (most errors near zero, fewer extreme errors).

### Why does it matter?
- If residuals aren't normally distributed, confidence intervals and hypothesis tests become unreliable.
- It affects the validity of statistical tests used to evaluate the model.

### How to check:
- Create a histogram of residuals and look for a bell-shaped curve.
- Use a **Q-Q Plot**: If points follow a straight diagonal line, residuals are normally distributed.

### Note:
- With large sample sizes (n > 30), this assumption becomes less critical due to the Central Limit Theorem.

---

## 4. **Homoscedasticity** (Equal Variance of Residuals)

The **variance** (spread) of residuals should be **constant** across all predicted values.

### In simple terms:
- Imagine throwing darts at different parts of a target.
- Homoscedasticity means the spread of your misses is the same everywhere, not tighter in some areas and wider in others.

### Why does it matter?
- If variance isn't constant, some predictions will be more reliable than others.
- This violates a key assumption and makes confidence intervals inaccurate.

### How to check:
- Plot predicted values (y_pred) on the x-axis and residuals on the y-axis.
- Look for a random, even spread of points.
- If you see a fan or funnel pattern, there's heteroscedasticity (unequal variance).

### How to fix:
- Apply transformations to the target variable (e.g., log transformation).
- Use weighted least squares regression.
- Include more relevant features.

---

## 5. **Autocorrelation of Residuals**

The **residuals should be independent** of each other—a residual at one point should not depend on residuals at other points.

### In simple terms:
- If you have data collected over time, the error in one time period shouldn't influence the error in the next time period.
- For example, if actual sales were higher than predicted in January, they shouldn't systematically be higher than predicted in February because of that.

### Why does it matter?
- Autocorrelation means the residuals are not truly random.
- This reduces the reliability of confidence intervals and statistical tests.
- The model is missing important patterns (like time-based trends).

### How to check:
- Plot residuals in order (usually by time) and look for patterns.
- If you see waves or trends, there's autocorrelation.
- Use the **Durbin-Watson statistic**: Values near 2 indicate no autocorrelation.

### How to fix:
- Include time-based features (e.g., lag variables).
- Use time series models (like ARIMA) if data has temporal patterns.
- Add missing variables that explain the autocorrelation.

---

## Summary

| Assumption | What to Check | Why It Matters |
|-----------|--------------|----------------|
| **Linearity** | Scatter plots (features vs. target) | Model assumes straight-line relationship |
| **No Multicollinearity** | VIF, correlation matrix | Features should be independent |
| **Normality of Residuals** | Histogram, Q-Q plot | Ensures valid statistical tests |
| **Homoscedasticity** | Residual plot | Variance should be constant |
| **No Autocorrelation** | Residual time plot, Durbin-Watson | Errors should be independent |

All five assumptions should be validated to build a robust and reliable linear regression model!

---
