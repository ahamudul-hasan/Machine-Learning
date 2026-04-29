# Naive Bayes Classifier

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Core Concepts](#core-concepts)
4. [Types of Naive Bayes](#types-of-naive-bayes)
5. [How It Works](#how-it-works)
6. [Advantages and Disadvantages](#advantages-and-disadvantages)
7. [Applications](#applications)
8. [Implementation Example](#implementation-example)

---

## Introduction

**Naive Bayes** is a probabilistic classification algorithm based on Bayes' theorem with a strong assumption of independence between features. Despite this "naive" assumption, it performs surprisingly well in many real-world applications and is particularly popular for text classification, sentiment analysis, and spam detection.

The algorithm is called "naive" because it assumes that all features are conditionally independent given the class label, which is rarely true in practice but simplifies the computation significantly.

---

## Mathematical Foundation

### Bayes' Theorem

The foundation of Naive Bayes is **Bayes' theorem**, which describes the probability of an event based on prior knowledge of conditions that might be related to the event:

$$P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}$$

Where:
- **$P(C|X)$** = Posterior probability (probability of class C given features X)
- **$P(X|C)$** = Likelihood (probability of features X given class C)
- **$P(C)$** = Prior probability (probability of class C)
- **$P(X)$** = Evidence/Marginal probability (probability of observing features X)

### Breaking Down the Components

#### 1. Prior Probability: $P(C)$

The prior probability represents our initial belief about the likelihood of each class before seeing any data:

$$P(C_k) = \frac{\text{Number of samples in class } C_k}{\text{Total number of samples}}$$

For example, if you have 100 emails and 25 are spam:
$$P(\text{Spam}) = \frac{25}{100} = 0.25$$
$$P(\text{Not Spam}) = \frac{75}{100} = 0.75$$

#### 2. Likelihood: $P(X|C)$

The likelihood is the probability of observing the features given that we know the class. For a single feature, this is straightforward, but with multiple features, we apply the **naive assumption**:

$$P(X_1, X_2, ..., X_n | C) = P(X_1|C) \cdot P(X_2|C) \cdot ... \cdot P(X_n|C) = \prod_{i=1}^{n} P(X_i|C)$$

This independence assumption is what makes the algorithm "naive." In reality, features are often correlated, but assuming independence dramatically reduces computational complexity from exponential to linear.

#### 3. Evidence: $P(X)$

The evidence is the total probability of observing the feature set across all classes:

$$P(X) = \sum_{k=1}^{K} P(X|C_k) \cdot P(C_k)$$

Where K is the number of classes.

### Final Classification Rule

For classification, we compute the posterior probability for each class and select the class with the highest probability:

$$\hat{C} = \arg\max_{C_k} P(C_k|X) = \arg\max_{C_k} \frac{P(X|C_k) \cdot P(C_k)}{P(X)}$$

Since $P(X)$ is constant for all classes, we can simplify:

$$\hat{C} = \arg\max_{C_k} P(X|C_k) \cdot P(C_k)$$

---

## Core Concepts

### 1. Conditional Independence Assumption

The naive assumption states that given the class, all features are independent:

$$P(X_i, X_j | C) = P(X_i|C) \cdot P(X_j|C) \text{ for } i \neq j$$

This is almost never true in real data, but it:
- Simplifies computation
- Reduces the need for large datasets
- Makes the algorithm scalable

### 2. Log-Space Computation

To avoid numerical underflow (very small probability numbers), we often work with log probabilities:

$$\log P(C|X) \propto \log P(C) + \sum_{i=1}^{n} \log P(X_i|C)$$

This transforms products into sums, which are numerically more stable.

### 3. Laplace Smoothing

When a feature-class combination doesn't appear in the training data, we'd get zero probability, which would zero out the entire product. To handle this, we add a small smoothing constant:

$$P(X_i = x|C) = \frac{\text{count}(X_i = x, C) + \alpha}{\text{count}(C) + \alpha \cdot |V|}$$

Where:
- $\alpha$ = smoothing parameter (usually 1)
- $|V|$ = size of vocabulary/unique values

---

## Types of Naive Bayes

### 1. Gaussian Naive Bayes

Used when features are continuous and follow a normal distribution.

$$P(X_i|C_k) = \frac{1}{\sqrt{2\pi\sigma_{k,i}^2}} \exp\left(-\frac{(X_i - \mu_{k,i})^2}{2\sigma_{k,i}^2}\right)$$

Where:
- $\mu_{k,i}$ = mean of feature $X_i$ in class $C_k$
- $\sigma_{k,i}^2$ = variance of feature $X_i$ in class $C_k$

**Use Case:** Predicting continuous values (e.g., iris flower classification)

### 2. Multinomial Naive Bayes

Used for discrete count data, typically in text classification where features represent word frequencies.

$$P(X_i|C_k) = \frac{count(X_i, C_k) + \alpha}{\sum_{j} count(X_j, C_k) + \alpha \cdot |V|}$$

**Use Case:** Spam detection, sentiment analysis, document classification

### 3. Bernoulli Naive Bayes

Used when features are binary (presence/absence of features).

$$P(X_i|C_k) = P_i \cdot X_i + (1 - P_i) \cdot (1 - X_i)$$

Where $P_i$ is the probability that feature $i$ is present in class $C_k$.

**Use Case:** Text classification with binary word presence, disease diagnosis

---

## How It Works

### Step-by-Step Process

#### Step 1: Calculate Prior Probabilities
For each class $C_k$, calculate:
$$P(C_k) = \frac{\text{count}(C_k)}{N}$$

#### Step 2: Calculate Likelihoods
For each feature $X_i$ and class $C_k$, calculate $P(X_i|C_k)$ using the appropriate distribution (Gaussian, Multinomial, or Bernoulli).

#### Step 3: Apply Bayes' Theorem
For a new sample with features $(X_1, X_2, ..., X_n)$, calculate:
$$P(C_k|X_1, X_2, ..., X_n) \propto P(C_k) \cdot \prod_{i=1}^{n} P(X_i|C_k)$$

#### Step 4: Select the Class
Choose the class with the maximum posterior probability:
$$\hat{C} = \arg\max_{C_k} P(C_k|X_1, X_2, ..., X_n)$$

### Example: Email Spam Classification

**Training Data:**
| Email | Words | Class |
|-------|-------|-------|
| "Buy now" | Buy, now | Spam |
| "Hello friend" | Hello, friend | Ham |
| "Cheap products" | Cheap, products | Spam |
| "How are you" | How, are, you | Ham |

**Priors:**
- $P(\text{Spam}) = 0.5$
- $P(\text{Ham}) = 0.5$

**Likelihoods (word frequencies):**
- $P(\text{"Buy"}|\text{Spam}) = \frac{1}{2} = 0.5$
- $P(\text{"Buy"}|\text{Ham}) = 0$
- $P(\text{"Hello"}|\text{Ham}) = \frac{1}{2} = 0.5$
- $P(\text{"Hello"}|\text{Spam}) = 0$

**Prediction for "Buy Hello":**
- $P(\text{Spam} | \text{"Buy"}, \text{"Hello"}) \propto 0.5 \times 0.5 \times 0 = 0$ (Contains "Hello" which is ham)
- $P(\text{Ham} | \text{"Buy"}, \text{"Hello"}) \propto 0.5 \times 0 \times 0.5 = 0$ (Contains "Buy" which is spam)

With smoothing (add-one):
- $P(\text{Spam} | \text{"Buy"}, \text{"Hello"}) \propto 0.5 \times \frac{2}{5} \times \frac{1}{5} = 0.04$
- $P(\text{Ham} | \text{"Buy"}, \text{"Hello"}) \propto 0.5 \times \frac{1}{5} \times \frac{2}{5} = 0.04$

---

## Advantages and Disadvantages

### Advantages ✓

1. **Simple and Fast**: Easy to understand and implement with minimal computational requirements
2. **Works with High-Dimensional Data**: Scales well to large feature spaces (e.g., text with thousands of words)
3. **Requires Small Training Sets**: Performs reasonably well even with limited training data
4. **Interpretable**: Probabilities are easy to interpret and explain
5. **Handles Irrelevant Features**: Often robust to irrelevant features
6. **Real-time Predictions**: Very fast prediction time

### Disadvantages ✗

1. **Strong Independence Assumption**: The conditional independence assumption rarely holds in practice
2. **Feature Scaling**: Sensitive to feature scaling in some variants
3. **Zero Frequency Problem**: Requires smoothing to handle unseen feature combinations
4. **Continuous Features**: Gaussian assumption may not fit all continuous distributions
5. **Correlated Features**: Performance degrades when features are highly correlated
6. **Data Imbalance**: Can be biased toward majority class

---

## Applications

### 1. **Text Classification**
- Spam detection
- Sentiment analysis
- Topic categorization
- Document classification

### 2. **Medical Diagnosis**
- Disease prediction
- Patient risk assessment
- Medical image classification

### 3. **Content Filtering**
- Email filtering
- Offensive content detection
- Recommendation systems

### 4. **Real-time Prediction**
- Fraud detection
- Credit approval
- Traffic classification

---

## Implementation Example

### Python with Scikit-learn

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Example 1: Gaussian Naive Bayes (Continuous Features)
from sklearn.datasets import iris

X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print(f"Gaussian NB Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Prior Probabilities: {gnb.class_prior_}")
print(f"Feature means per class:\n{gnb.theta_}")
print(f"Feature variances per class:\n{gnb.var_}")

# Example 2: Multinomial Naive Bayes (Text Classification)
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "Python is great for data science",
    "Machine learning is fascinating",
    "Python machine learning",
    "Data science with Python"
]
labels = [1, 1, 1, 1]  # All positive

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

mnb = MultinomialNB()
mnb.fit(X, labels)

# Predict new document
new_doc = vectorizer.transform(["Python data science"])
prediction = mnb.predict(new_doc)
probability = mnb.predict_proba(new_doc)

print(f"\nPrediction: {prediction}")
print(f"Probability: {probability}")

# Example 3: Bernoulli Naive Bayes (Binary Features)
bnb = BernoulliNB()
X_binary = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
y_binary = np.array([1, 0, 1, 0])

bnb.fit(X_binary, y_binary)
print(f"\nBernoulli NB prediction: {bnb.predict([[1, 1]])}")
```

### Manual Implementation

```python
import numpy as np

class NaiveBayesGaussian:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.parameters[c] = {
                'mean': X_c.mean(axis=0),
                'var': X_c.var(axis=0),
                'priors': len(X_c) / len(X)
            }
    
    def _calculate_likelihood(self, x, mean, var):
        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.parameters[c]['priors'])
                likelihood = np.sum(
                    np.log(self._calculate_likelihood(
                        x,
                        self.parameters[c]['mean'],
                        self.parameters[c]['var']
                    ))
                )
                posteriors.append(prior + likelihood)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)
```

---

## Summary

Naive Bayes is a powerful and practical classification algorithm that, despite its simplistic assumptions, provides excellent results in many domains. Its simplicity, efficiency, and ability to work with high-dimensional data make it an excellent choice for many classification problems, particularly in text analysis and real-time prediction scenarios.

The key to successful implementation lies in:
1. Understanding the type of data (continuous, discrete, binary)
2. Choosing the appropriate Naive Bayes variant
3. Properly handling feature preprocessing and smoothing
4. Evaluating performance on diverse test sets

---
