## 1. Scikit learn train test split:

train test split is a function in **Scikit-learn** that helps you **divide your dataset** into two parts:

* **Training data** → used to **teach** your model.
* **Testing data** → used to **check** how well your model learned.


If you train and test your model on the  **same data** , it might just **memorize** the answers — not actually  **learn patterns** . So we keep some data aside (test data) to see how it performs on  **new, unseen data** .

Code:

{from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)}

* x → your input features (for example, columns like “age”, “income”, etc.)
* y → your target/output (for example, “will_buy” = yes/no)
* test_size=0.2 → means 20% of data is for testing, 80% for training.
* random_state=42 → ensures the split is **the same every time** you run the code (for consistency).


## 2. One Hot Encoding

It’s a way to **convert words (categories)** into **numbers** so that a machine learning model can understand them.

Code:

encoder = OneHotEncoder(sparse_output=False)

We create an object called encoder to do the one-hot encoding.

* sparse_output=False means:

  --Give the result as a **normal table (array)** instead of a compressed format.
