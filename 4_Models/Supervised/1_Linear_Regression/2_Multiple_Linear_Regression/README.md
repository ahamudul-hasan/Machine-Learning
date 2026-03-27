# Multiple Linear Regression - Theory & Mathematical Explanation

## What is Multiple Linear Regression?

Multiple Linear Regression is a statistical method used to predict a continuous target variable based on **multiple independent variables** (features). It's an extension of Simple Linear Regression, which uses only one feature.

**Simple Analogy:** If Simple Linear Regression is like predicting house price based on only size, Multiple Linear Regression is like predicting house price based on size, location, age, number of rooms, etc.

---

## Mathematical Formula

The equation for Multiple Linear Regression is:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + ... + \beta_n x_n + \epsilon$$

Where:
- **y** = Target variable (what we're trying to predict)
- **x₁, x₂, x₃, ..., xₙ** = Independent variables (features/input variables)
- **β₀** = Intercept (constant term) - where the line crosses the y-axis
- **β₁, β₂, β₃, ..., βₙ** = Coefficients (slopes) - how much each feature affects the target
- **ε** = Error term (the difference between actual and predicted values)

### Simple Example:
Predicting house price with 2 features:

$$Price = \beta_0 + \beta_1 \times Size + \beta_2 \times Bedrooms + \epsilon$$

If β₀ = 50,000, β₁ = 200, β₂ = 10,000:

$$Price = 50,000 + 200 \times Size + 10,000 \times Bedrooms$$

This means:
- Base price starts at $50,000
- Each square foot adds $200
- Each bedroom adds $10,000

---

## How Does It Work?

### Step 1: Fit the Model
The algorithm finds the best values for β₀, β₁, β₂, ..., βₙ that minimize the error between actual and predicted values.

**Goal:** Minimize the sum of squared errors (Least Squares Method)

$$\text{SSE} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Where:
- **yᵢ** = Actual value
- **ŷᵢ** = Predicted value

### Step 2: Make Predictions
Once coefficients are found, use the equation to predict new values.

---

## Mathematical Formulation From Scratch

### The Prediction Equation (Matrix Form)

We can represent all predictions at once using matrix notation:

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{\beta}$$

Where:
- **ŷ** = Predicted values (n × 1 matrix)
- **X** = Feature matrix (n × (m+1)) - includes constant column of 1s for intercept
- **β** = Coefficient vector ((m+1) × 1)
- **n** = Number of samples
- **m** = Number of features

**Example with 2 features and 3 samples:**

$$\begin{bmatrix} \hat{y}_1 \\ \hat{y}_2 \\ \hat{y}_3 \end{bmatrix} = \begin{bmatrix} 1 & x_{1,1} & x_{1,2} \\ 1 & x_{2,1} & x_{2,2} \\ 1 & x_{3,1} & x_{3,2} \end{bmatrix} \begin{bmatrix} \beta_0 \\ \beta_1 \\ \beta_2 \end{bmatrix}$$

---

### The Cost Function (Loss Function)

We want to minimize the **Sum of Squared Errors (SSE)**:

$$J(\mathbf{\beta}) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

In matrix form:

$$J(\mathbf{\beta}) = (\mathbf{y} - \mathbf{X}\mathbf{\beta})^T(\mathbf{y} - \mathbf{X}\mathbf{\beta})$$

Or expanded:

$$J(\mathbf{\beta}) = \mathbf{y}^T\mathbf{y} - 2\mathbf{\beta}^T\mathbf{X}^T\mathbf{y} + \mathbf{\beta}^T\mathbf{X}^T\mathbf{X}\mathbf{\beta}$$

---

### Finding the Best Coefficients (Least Squares Solution)

To find the minimum of the cost function, we take the derivative with respect to **β** and set it to zero:

$$\frac{\partial J(\mathbf{\beta})}{\partial \mathbf{\beta}} = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\mathbf{\beta} = 0$$

Solving for **β**:

$$2\mathbf{X}^T\mathbf{X}\mathbf{\beta} = 2\mathbf{X}^T\mathbf{y}$$

$$\mathbf{X}^T\mathbf{X}\mathbf{\beta} = \mathbf{X}^T\mathbf{y}$$

### **Normal Equation (Closed-Form Solution):**

$$\mathbf{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

This is the solution that minimizes the error. We directly calculate the optimal coefficients without iteration!

---

### Step-by-Step Example

**Given Data:**
- 3 samples with 1 feature and target

| X₁ | y |
|----|---|
| 1  | 3 |
| 2  | 5 |
| 3  | 7 |

**Step 1: Build Feature Matrix X (add column of 1s)**

$$\mathbf{X} = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} 3 \\ 5 \\ 7 \end{bmatrix}$$

**Step 2: Calculate X^T X**

$$\mathbf{X}^T\mathbf{X} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix} = \begin{bmatrix} 3 & 6 \\ 6 & 14 \end{bmatrix}$$

**Step 3: Calculate X^T y**

$$\mathbf{X}^T\mathbf{y} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 3 \\ 5 \\ 7 \end{bmatrix} = \begin{bmatrix} 15 \\ 36 \end{bmatrix}$$

**Step 4: Calculate (X^T X)^(-1)**

$$(X^T X)^{-1} = \begin{bmatrix} 3 & 6 \\ 6 & 14 \end{bmatrix}^{-1} = \begin{bmatrix} 1.4 & -0.6 \\ -0.6 & 0.3 \end{bmatrix}$$

**Step 5: Calculate β using Normal Equation**

$$\mathbf{\beta} = \begin{bmatrix} 1.4 & -0.6 \\ -0.6 & 0.3 \end{bmatrix} \begin{bmatrix} 15 \\ 36 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$

**Result:** β₀ = 1 (intercept), β₁ = 2 (slope)

**Equation:** y = 1 + 2x

---

### Alternative: Gradient Descent

While the Normal Equation gives us the exact solution, sometimes we use **Gradient Descent** for large datasets:

1. Start with random β values
2. Calculate the gradient (direction of steepest increase)
3. Move in opposite direction (downhill)
4. Repeat until convergence

**Gradient formula:**

$$\frac{\partial J}{\partial \beta_j} = -2 \sum_{i=1}^{n} x_{i,j}(y_i - \hat{y}_i)$$

**Update rule:**

$$\beta_j := \beta_j - \alpha \frac{\partial J}{\partial \beta_j}$$

Where **α** is the learning rate (how big a step we take).

---

### Normal Equation vs Gradient Descent

| Aspect | Normal Equation | Gradient Descent |
|--------|-----------------|------------------|
| **Computation** | Closed-form (direct) | Iterative |
| **Time Complexity** | O(m³) - slower for many features | O(n×m) per iteration |
| **Space** | Requires matrix inversion | Less memory intensive |
| **Best for** | Small to medium datasets (m < 10k) | Large datasets |
| **Guarantee** | Exact solution (if invertible) | Approximate (needs tuning) |

---

## Key Concepts

### 1. **Linear Relationship**
Multiple Linear Regression assumes a **linear relationship** between features and target. This means changes in features cause proportional changes in the target.

### 2. **Coefficients (β values)**
- **Positive coefficient** = Feature increases → Target increases
- **Negative coefficient** = Feature increases → Target decreases
- **Zero coefficient** = Feature has no effect on target

### 3. **Intercept (β₀)**
The predicted value when all features are zero.

### 4. **Residuals/Errors**
The differences between actual and predicted values. A good model has small residuals.

---

## Advantages

✅ Simple to understand and implement  
✅ Computationally efficient  
✅ Works well for linear relationships  
✅ Provides interpretable coefficients  
✅ Fast training and prediction  

---

## Limitations

❌ Assumes linear relationship between features and target  
❌ Sensitive to outliers  
❌ May not work well with non-linear data  
❌ Multicollinearity (high correlation between features) can be problematic  

---

## Real-World Example

**Predicting Student Performance:**

$$GPA = 3.0 + 0.5 \times StudyHours + 0.1 \times Attendance + (-0.3) \times SkippedClasses$$

- Base GPA: 3.0
- Each study hour adds 0.5 to GPA
- Each attendance percentage point adds 0.1
- Each skipped class reduces GPA by 0.3

---

## Model Evaluation Metrics

After training the model, we evaluate it using:

### 1. **Mean Absolute Error (MAE)**
Average absolute difference between actual and predicted values.

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### 2. **Mean Squared Error (MSE)**
Average of squared differences. Penalizes larger errors more.

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### 3. **R² Score (Coefficient of Determination)**
How well the model explains the variance in data (0 to 1, higher is better).

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

Where:
- **SSₓₑₛ** = Sum of squared residuals (prediction errors)
- **SStot** = Total sum of squares (variance in actual data)

---

## Summary

Multiple Linear Regression is a foundational machine learning algorithm that:
- Uses multiple features to predict a continuous target
- Finds the best-fitting straight line (hyperplane) through data
- Is interpretable and computationally efficient
- Works best when data has linear relationships

It's the starting point for understanding regression and serves as a baseline to compare more complex models.

