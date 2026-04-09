# Supervised and Unsupervised Learning (with Regression Context)

This guide explains what supervised learning is, what unsupervised learning is, and where regression belongs.

## 1) What is Supervised Learning?

Supervised learning means:
- You train a model using input data (features) and correct output labels (targets).
- The model learns a mapping from X -> y.
- Goal: predict the target for new, unseen data.

Common supervised tasks:
- Regression: predict continuous values (example: house price, salary, temperature).
- Classification: predict categories (example: spam/not spam, disease/no disease).

## 2) What is Unsupervised Learning?

Unsupervised learning means:
- You train a model using input data only (no target labels).
- The model tries to find hidden structure, patterns, or groups.
- Goal: understand data distribution and relationships.

Common unsupervised tasks:
- Clustering (example: customer segmentation).
- Dimensionality reduction (example: PCA for visualization or compression).
- Association pattern mining.

## 3) Is Regression Supervised or Unsupervised?

Regression is a supervised learning task.
- Why: regression needs a target value y during training.
- The model minimizes error between predicted and true numeric targets.

So, the standard term is:
- Supervised regression

## 4) What About "Unsupervised Regression"?

In standard machine learning terminology, "unsupervised regression" is not a common formal task.

People sometimes use this phrase informally when they mean:
- Finding trends or structure in unlabeled data.
- Reducing dimensions and then fitting simple trends.
- Self-supervised or semi-supervised setups (different from pure unsupervised learning).

But in core ML courses and interviews:
- Regression = supervised learning.
- Unsupervised learning = clustering, dimensionality reduction, etc.

## 5) Quick Comparison

| Aspect | Supervised Learning | Unsupervised Learning |
|---|---|---|
| Labels available? | Yes (target y) | No |
| Main goal | Predict output | Discover structure |
| Typical output | Value/class prediction | Clusters/components/patterns |
| Example algorithms | Linear Regression, Logistic Regression, Random Forest | K-Means, PCA, DBSCAN |

## 6) Examples from Real Life

- Supervised regression: predict house price from area, location, and rooms.
- Supervised classification: predict whether a student will pass/fail.
- Unsupervised clustering: group customers by purchase behavior.
- Unsupervised PCA: reduce 100 features to 2 features for visualization.

## Final Takeaway

- Supervised learning uses labeled data.
- Unsupervised learning uses unlabeled data.
- Regression belongs to supervised learning.
