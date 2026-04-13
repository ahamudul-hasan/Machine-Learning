# Classification Metrics: Accuracy & Confusion Matrix

## Table of Contents
- [Overview](#overview)
- [1. Accuracy](#1-accuracy)
  - [What is Accuracy?](#what-is-accuracy)
  - [Formula](#formula)
  - [Visual Explanation](#visual-explanation)
  - [Simple Example](#simple-example)
  - [When to Use Accuracy](#when-to-use-accuracy)
  - [When NOT to Use Accuracy](#when-not-to-use-accuracy)
- [When Accuracy is Misleading ⚠️](#when-accuracy-is-misleading-️)
  - [The Problem with Accuracy](#the-problem-with-accuracy)
  - [Real Example: The Useless Model Problem](#real-example-the-useless-model-problem)
  - [Why is This Happening?](#why-is-this-happening)
  - [The Real Problem](#the-real-problem)
  - [Another Example: Spam Detection](#another-example-spam-detection)
  - [When Accuracy is Most Misleading](#when-accuracy-is-most-misleading)
  - [How to Spot If Accuracy is Misleading](#how-to-spot-if-accuracy-is-misleading)
  - [The Solution: Don't Use Accuracy Alone!](#the-solution-dont-use-accuracy-alone)
  - [Key Takeaway](#key-takeaway)
- [2. Confusion Matrix](#2-confusion-matrix)
  - [What is a Confusion Matrix?](#what-is-a-confusion-matrix)
  - [Breaking Down Each Cell](#breaking-down-each-cell)
  - [Real Example](#real-example)
  - [Useful Metrics Derived from Confusion Matrix](#useful-metrics-derived-from-confusion-matrix)
  - [When Confusion Matrix is Useful](#when-confusion-matrix-is-useful)
- [3. Type 1 and Type 2 Errors](#3-type-1-and-type-2-errors)
  - [What Are Type 1 and Type 2 Errors?](#what-are-type-1-and-type-2-errors)
  - [Type 1 Error (False Positive)](#type-1-error-false-positive)
  - [Type 2 Error (False Negative)](#type-2-error-false-negative)
  - [Visual Comparison](#visual-comparison)
  - [Which Error is Worse?](#which-error-is-worse)
  - [Example with Numbers](#example-with-numbers)
  - [Controlling Type 1 and Type 2 Errors](#controlling-type-1-and-type-2-errors)
  - [Example: Email Spam Filter](#example-email-spam-filter)
- [Comparison: Accuracy vs Confusion Matrix](#comparison-accuracy-vs-confusion-matrix)

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

## When Accuracy is Misleading ⚠️

### The Problem with Accuracy

Accuracy can **trick you into thinking your model is great when it's actually terrible**. This happens especially with **imbalanced datasets**.

### Real Example: The Useless Model Problem

Imagine you build a disease detection model with this dataset:
- Total patients: **1000**
- Actually have disease: **10** (1%)
- Actually healthy: **990** (99%)

#### Scenario 1: Your "Smart" Model

Your model:
- Correctly identifies 8 sick patients (TP = 8)
- Correctly identifies 980 healthy people (TN = 980)
- Misses 2 sick patients (FN = 2)
- False alarms for 10 healthy people (FP = 10)

```
Accuracy = (8 + 980) / 1000 = 988 / 1000 = 98.8%
```

**Wow! 98.8% accuracy! That seems great!**

#### Scenario 2: The Lazy "Useless" Model

But wait... what if your model is **completely dumb** and just predicts "Everyone is healthy" for every patient?

```
Prediction: Healthy for ALL 1000 patients

Accuracy = 990 / 1000 = 99%
```

**The useless model has 99% accuracy!** 😱

### Why is This Happening?

Because the dataset is **imbalanced**:
- 99% of patients are healthy
- 1% have the disease

When you just predict "Healthy" for everyone, you're automatically correct 99% of the time, **even though you're not catching any sick patients!**

### The Real Problem

The "useless" model has:
- ✓ 99% accuracy
- ✗ 0 sick patients detected (caught 0 out of 10)
- ✗ Completely useless for diagnosis

Your better model has:
- ✓ 98.8% accuracy (slightly lower)
- ✓ Actually catches 80% of sick patients
- ✓ Much more useful!

### Another Example: Spam Detection

Email dataset:
- Total emails: **10,000**
- Spam emails: **100** (1%)
- Legitimate emails: **9,900** (99%)

Model that just predicts "Legitimate" for everything:
```
Accuracy = 9,900 / 10,000 = 99%
```

**Your spam filter has 99% accuracy but catches ZERO spam emails!** 🚫

### When Accuracy is Most Misleading

| Scenario | Accuracy Shows | Reality Is |
|----------|---|---|
| **Imbalanced data** | Looks great (99%+) | Model may be useless |
| **One class dominates** | High numbers | Model ignores minority class |
| **Cost-sensitive problems** | Misleading | Missing rare cases is very costly |
| **Medical diagnosis** | Deceptive | Missing diseases kills people |
| **Fraud detection** | Deceptive | Missing fraud costs money |

### How to Spot If Accuracy is Misleading

Ask yourself these questions:

1. **Is my data imbalanced?**
   - Count: How many of each class do I have?
   - If one class is much larger (>70%), be careful with accuracy

2. **What happens if my model predicts the majority class only?**
   - Calculate: What would accuracy be?
   - If it's similar to your model's accuracy, that's a red flag!

3. **Which type of error is more costly?**
   - Missing a sick patient (Type 2 error) = Life-threatening
   - False alarm for healthy person (Type 1 error) = Worried person
   - These aren't equal in cost!

### The Solution: Don't Use Accuracy Alone!

When you have **imbalanced data** or **cost-sensitive problems**, use these metrics instead or in addition to accuracy:

1. **Precision**: "When model says positive, how often is it right?"
2. **Recall**: "Did we catch all the positive cases?"
3. **F1-Score**: Balanced combination of precision and recall
4. **Confusion Matrix**: See the real breakdown
5. **ROC-AUC**: Good for imbalanced data
6. **PR-AUC**: For highly imbalanced data

### Key Takeaway

**⚠️ Always check the confusion matrix before celebrating high accuracy!**

High accuracy + imbalanced data = Potential trap 🪤

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

## 3. Type 1 and Type 2 Errors

### What Are Type 1 and Type 2 Errors?

Type 1 and Type 2 errors come from **hypothesis testing** in statistics. These errors describe the two ways your model can be **wrong**:

### Type 1 Error (False Positive)

**Type 1 Error** = **False Positive (FP)**

- **What it is**: You **reject** something that is actually **true**. Your model says "YES" when the truth is "NO".
- **In simple terms**: A false alarm
- **Think of it as**: Incorrectly identifying something that isn't there

#### Real Examples:

1. **Medical Diagnosis**
   - Model says: "You have disease"
   - Reality: "You don't have disease"
   - Consequence: Unnecessary worry and treatment costs

2. **Spam Detection**
   - Model says: "This is spam"
   - Reality: "It's a legitimate email"
   - Consequence: You miss an important email

3. **Fraud Detection**
   - Model says: "This transaction is fraudulent"
   - Reality: "It's a legitimate transaction"
   - Consequence: Customer's card gets blocked unnecessarily

### Type 2 Error (False Negative)

**Type 2 Error** = **False Negative (FN)**

- **What it is**: You **fail to reject** something that is actually **false**. Your model says "NO" when the truth is "YES".
- **In simple terms**: A missed case
- **Think of it as**: Failing to identify something that is there

#### Real Examples:

1. **Medical Diagnosis**
   - Model says: "You don't have disease"
   - Reality: "You do have disease"
   - Consequence: Disease goes untreated (very dangerous!)

2. **Spam Detection**
   - Model says: "This is legitimate"
   - Reality: "It's actually spam"
   - Consequence: Spam reaches your inbox

3. **Fraud Detection**
   - Model says: "Transaction is legitimate"
   - Reality: "It's actually fraudulent"
   - Consequence: Fraud goes undetected (company loses money)

### Visual Comparison

| Error Type | What Happened | Prediction | Reality | Type | Severity Often |
|---|---|---|---|---|---|
| **Type 1 Error** | False Alarm | YES | NO | FP | Varies |
| **Type 2 Error** | Missed Case | NO | YES | FP | Often High |

### Which Error is Worse?

**It depends on your problem!**

#### When Type 1 Error (FP) is Worse:
- **Spam Filters**: False positives delete important emails
- **Fraud Alerts**: False alerts annoy customers
- **Ads**: Showing wrong ads wastes money
- **In general**: When false alarms are expensive or annoying

#### When Type 2 Error (FN) is Worse:
- **Medical Diagnosis**: Missing a disease can be life-threatening
- **Security**: Missing a terrorist threat is catastrophic
- **Defect Detection**: Missing a defective product harms customers
- **In general**: When missing something is dangerous or costly

### Example with Numbers

From our disease prediction example:

```
                  Predicted Healthy    Predicted Sick
Actually Healthy         85                  10  ← Type 1 Errors (FP)
Actually Sick             5  ← Type 2 Errors (FN)    80
```

- **Type 1 Errors = 10**: 10 healthy people incorrectly told they're sick
- **Type 2 Errors = 5**: 5 sick people incorrectly told they're healthy

### Controlling Type 1 and Type 2 Errors

You can adjust your model to reduce one type of error, but **increasing one usually increases the other**:

```
More Strict (Reduce FP)          More Lenient (Reduce FN)
├─ Higher threshold                ├─ Lower threshold
├─ Fewer false alarms             ├─ Catch more cases
├─ More Type 2 Errors             ├─ More Type 1 Errors
└─ Lower Recall, Higher Precision  └─ Higher Recall, Lower Precision
```

### Example: Email Spam Filter

If you make the filter **very strict**:
- ✓ Few spam emails (low FP) get through
- ✗ Many legitimate emails get blocked (high FN)

If you make the filter **very lenient**:
- ✓ Few legitimate emails get blocked (low FN)
- ✗ Many spam emails (high FP) get through


## Comparison: Accuracy vs Confusion Matrix

| Aspect | Accuracy | Confusion Matrix |
|--------|----------|------------------|
| **What it shows** | Single percentage | Breakdown of all predictions |
| **Detail level** | Summary only | Very detailed |
| **Best for** | Quick check | Deep understanding |
| **Works with imbalanced data?** | No | Yes |
| **Shows error types?** | No | Yes |

---

