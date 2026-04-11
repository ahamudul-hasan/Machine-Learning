# Logistic Regression

## Overview

Logistic Regression is a fundamental classification algorithm used for binary classification problems. Despite its name, it's a classification algorithm, not a regression algorithm. It models the probability of a binary outcome based on one or more predictor variables.

## What is Logistic Regression?

Logistic Regression estimates the probability that an instance belongs to a particular class. It uses the **logistic function** (sigmoid function) to transform linear combinations of input features into probabilities between 0 and 1.

## The Sigmoid Function

The sigmoid function is the core mathematical function that powers logistic regression. It transforms any real-valued number into a probability between 0 and 1.

### Mathematical Formula

The sigmoid function is defined as:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where:
- $z$ is the linear combination of features: $z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n$
- $e$ is Euler's number (approximately 2.718)
- The output is always between 0 and 1

### Key Properties

1. **Output Range**: Always produces values between 0 and 1, making it ideal for probability estimation
2. **Smooth and Differentiable**: The function is smooth everywhere, which is important for optimization
3. **Monotonicity**: It's monotonically increasing, so larger inputs always produce larger outputs
4. **Symmetry**: $\sigma(-z) = 1 - \sigma(z)$, symmetric around the point (0, 0.5)
5. **Decision Boundary**: When $\sigma(z) = 0.5$, we have $z = 0$ (the decision boundary)

### How It Works

- When $z \to +\infty$: $\sigma(z) \to 1$ (high probability of class 1)
- When $z \to -\infty$: $\sigma(z) \to 0$ (high probability of class 0)
- When $z = 0$: $\sigma(z) = 0.5$ (equal probability for both classes)

### Sigmoid Function Behavior

```
Probability
    1.0 |                    ╱╱╱╱
    0.9 |                ╱╱╱
    0.7 |            ╱╱╱
    0.5 |        ╱╱╱ ← Decision Boundary
    0.3 |    ╱╱╱
    0.1 |╱╱╱
    0.0 |________________________
       -5  -3  -1   0   1   3   5
               Linear Input (z)
```

### Why Use Sigmoid in Logistic Regression?

1. **Probability Interpretation**: The output naturally represents a probability
2. **Handles Non-linearity**: Transforms linear relationships into non-linear decision boundaries
3. **Gradient-Based Learning**: Its derivative is simple ($\sigma'(z) = \sigma(z)(1 - \sigma(z))$), making it efficient for gradient descent optimization
4. **Clear Decision Rule**: Easy interpretation - if $\sigma(z) > 0.5$, predict class 1; otherwise predict class 0

### Example

If we have a model where $z = 2x - 1$:
- When $x = 0$: $z = -1$, $\sigma(-1) \approx 0.27$ (27% probability of class 1)
- When $x = 1$: $z = 1$, $\sigma(1) \approx 0.73$ (73% probability of class 1)
- When $x = 0.5$: $z = 0$, $\sigma(0) = 0.5$ (50% probability, decision boundary)


## The Perceptron Trick

The Perceptron Trick is a foundational concept in machine learning that explains how classification algorithms adjust their decision boundaries to correctly classify points.

### How It Works

1. **Initialize**: Start with a random decision boundary
2. **For Each Misclassified Point**:
   - If the algorithm predicts the wrong class, the decision boundary needs to move
   - Move the boundary closer to the misclassified point
   - This is done by updating the weights in the direction that reduces the error

3. **Repeat**: Continue until all points are correctly classified (or a stopping criterion is met)

### Mathematical Intuition

For a misclassified point $(x, y)$:
- If the model predicts 0 but $y = 1$ (false negative): Move the boundary toward the point
- If the model predicts 1 but $y = 0$ (false positive): Move the boundary away from the point

The weight update rule:
$$\beta = \beta + \alpha \cdot (y - \hat{y}) \cdot x$$

Where:
- $\alpha$ is the learning rate
- $(y - \hat{y})$ is the prediction error
- $x$ is the feature vector

### Visual Representation

```
Initial Boundary (incorrect):
    Class 0: ○  ○
    Class 1: ●  ●  ●  ●
    -------- Boundary --------
    ○ (misclassified)

After Perceptron Trick:
    Class 0: ○  ○
    -------- Boundary --------
    Class 1: ●  ●  ●  ●
    ● (correctly classified now)
```

## Advantages

✓ Simple and interpretable  
✓ Computationally efficient  
✓ Works well for linearly separable data  
✓ Provides probability estimates  
✓ Good baseline model  

## Disadvantages

✗ Assumes linear relationship between features and output  
✗ Struggles with non-linearly separable data  
✗ Sensitive to outliers  
✗ Requires feature scaling for optimal performance  
✗ Binary classification only (extension to multi-class exists)  

## When to Use Logistic Regression

- **Medical Diagnosis**: Predict if a patient has a disease (0 or 1)
- **Email Spam Detection**: Classify emails as spam or not spam
- **Customer Churn**: Predict if a customer will leave
- **Credit Approval**: Decide whether to approve a loan
- **Baseline Model**: Start with logistic regression before trying complex models

## Implementation Steps

1. **Load and Explore Data**: Understand the dataset
2. **Preprocess Data**: Handle missing values, scale features
3. **Split Data**: Divide into training and testing sets
4. **Train Model**: Fit logistic regression
5. **Evaluate**: Use metrics like accuracy, precision, recall, F1-score, AUC-ROC
6. **Predict**: Make predictions on new data

---

