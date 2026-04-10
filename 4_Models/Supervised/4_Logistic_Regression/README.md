# Logistic Regression

## Overview

Logistic Regression is a fundamental classification algorithm used for binary classification problems. Despite its name, it's a classification algorithm, not a regression algorithm. It models the probability of a binary outcome based on one or more predictor variables.

## What is Logistic Regression?

Logistic Regression estimates the probability that an instance belongs to a particular class. It uses the **logistic function** (sigmoid function) to transform linear combinations of input features into probabilities between 0 and 1.


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

