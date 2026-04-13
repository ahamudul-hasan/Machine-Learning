# Classification Metrics: Accuracy & Confusion Matrix

## Overview
When we build classification models (models that predict categories like Yes/No, Pass/Fail, etc.), we need ways to measure how well they perform. The two fundamental metrics for this are **Accuracy** and **Confusion Matrix**.

---

## 1. Accuracy

### What is Accuracy?

**Accuracy** is the simplest metric to understand. It tells you: **Out of all your predictions, how many were correct?**

### Formula
```
Accuracy = Number of Correct Predictions / Total Number of Predictions
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Visual Explanation

![Accuracy Concept](../../image/accuracy_concept.png)

### Simple Example

Imagine a weather prediction model makes 100 predictions:
- ✓ **Correct predictions: 80**
- ✗ **Wrong predictions: 20**

Then: **Accuracy = 80/100 = 80%**

This means the model is correct 8 out of 10 times.

### When to Use Accuracy
- ✓ When all classes (categories) are equally important
- ✓ When the dataset is balanced (similar number of each class)
- ✓ For quick overall model performance check

### When NOT to Use Accuracy
- ✗ When classes are imbalanced (e.g., 95% healthy, 5% sick)
- ✗ When one type of error is more costly than another
- ✗ When you need to understand different types of mistakes

---

## 2. Confusion Matrix

### What is a Confusion Matrix?

A **Confusion Matrix** is a table that shows **in detail** what types of mistakes your model makes. Instead of just giving you one number (like accuracy), it breaks down predictions into 4 categories:

#### For Binary Classification (2 classes):

| | Predicted: No | Predicted: Yes |
|---|---|---|
| **Actually: No** | TN (True Negative) | FP (False Positive) |
| **Actually: Yes** | FN (False Negative) | TP (True Positive) |

### Breaking Down Each Cell

1. **TP (True Positive)** - Model predicted **Yes**, Actually **Yes**
   - ✓ CORRECT! The model caught the positive case.
   - Example: Predicted disease, patient has disease.

2. **TN (True Negative)** - Model predicted **No**, Actually **No**
   - ✓ CORRECT! The model correctly identified a negative case.
   - Example: Predicted no disease, patient is healthy.

3. **FP (False Positive)** - Model predicted **Yes**, Actually **No**
   - ✗ WRONG! False alarm.
   - Example: Predicted disease, but patient is healthy.

4. **FN (False Negative)** - Model predicted **No**, Actually **Yes**
   - ✗ WRONG! Missed the case.
   - Example: Predicted no disease, but patient has disease.

### Real Example

![Confusion Matrix Example](../../image/confusion_matrix_example.png)

#### Disease Prediction Example:

```
Predicted →
Actual ↓       |  Healthy (No)  |  Sick (Yes)  |
─────────────────────────────────────────────────
Healthy (No)   |      85        |      10      |
─────────────────────────────────────────────────
Sick (Yes)     |       5        |      80      |
─────────────────────────────────────────────────
```

- **TP = 80**: Model correctly identified 80 sick patients
- **TN = 85**: Model correctly identified 85 healthy people
- **FP = 10**: Model said 10 people were sick (but they're healthy) - False alarm
- **FN = 5**: Model said 5 people were healthy (but they're sick) - Missed cases

### Useful Metrics Derived from Confusion Matrix

Once you have the confusion matrix, you can calculate many other metrics:

1. **Accuracy** = (TP + TN) / Total
   - For example: (80 + 85) / 180 = 91.7%

2. **Precision** = TP / (TP + FP)
   - Of all cases we predicted as positive, how many were actually positive?
   - Answers: "When the model says YES, how often is it right?"

3. **Recall (Sensitivity)** = TP / (TP + FN)
   - Of all actual positive cases, how many did we catch?
   - Answers: "Did we find all the positive cases?"

4. **Specificity** = TN / (TN + FP)
   - Of all actual negative cases, how many did we correctly identify?
   - Answers: "Did we correctly reject negative cases?"

5. **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
   - Balances precision and recall when they're both important

### When Confusion Matrix is Useful

- ✓ Understanding what types of mistakes the model makes
- ✓ Imbalanced datasets (different number of classes)
- ✓ When different errors have different costs
- ✓ Tuning the model for specific priorities (catch more positives vs. fewer false alarms)

---

## Comparison: Accuracy vs Confusion Matrix

| Aspect | Accuracy | Confusion Matrix |
|--------|----------|------------------|
| **What it shows** | Single percentage | Breakdown of all predictions |
| **Detail level** | Summary only | Very detailed |
| **Best for** | Quick check | Deep understanding |
| **Works with imbalanced data?** | No | Yes |
| **Shows error types?** | No | Yes |

---

