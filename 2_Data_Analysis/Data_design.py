import pandas as pd
import matplotlib.pyplot as mat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions

df = pd.read_csv('Data/placement.csv')

print(df)

# df = df.iloc[:,1:]

# print(df)

# mat.scatter(df['cgpa'],df['iq'],c=df['placement'])
# mat.show()

x = df.iloc[:,0:2]
y = df.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)

clf = LogisticRegression()



print(plot_decision_regions(x_train,y_train.values, clf=clf, legend=2))

