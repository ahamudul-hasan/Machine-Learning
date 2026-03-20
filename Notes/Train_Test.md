# 📊 Understanding X and y in Machine Learning

## 🔹 What are X and y?

In machine learning:

- **X** = Input data (features)
- **y** = Output (target / label)

👉 Simple:
> X = what you give to the model  
> y = what you want the model to predict  

---

## 🔹 Example

Let’s say we want to predict house prices:

| Size (X) | Price (y) |
|----------|----------|
| 1000 sqft | 200k |
| 1500 sqft | 300k |

- **X = Size**
- **y = Price**

---

## 🔹 Why do we split into Train and Test?

We split data like this:

```python
x_train, x_test, y_train, y_test
```

### Meaning:

- **x_train** → Data used to train the model  
- **y_train** → Correct answers for training  

👉 Model learns:
```
X → y
```

- **x_test** → New unseen data  
- **y_test** → Real answers to check accuracy  

---

## 🔹 Why separate X and y?

Because machine learning works like this:

> “Given X, predict y”

So we must keep:
- Inputs (X)
- Outputs (y)

separate.

---

## 🔹 Simple Analogy 🎓

Think of it like an exam:

- **X = Questions**
- **y = Answers**

### Training:
- You study questions + answers

### Testing:
- You get only questions (X_test)
- You predict answers
- Then compare with real answers (y_test)

---

## 🔹 Code Example

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

---

## 🔹 Final Idea

- **X = what you know**
- **y = what you want to predict**

---

## ✅ Summary

- X = Input features  
- y = Target/output  
- Train set = Learning  
- Test set = Evaluation  

👉 This is the basic foundation of all machine learning models.