import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.linear_model.tests.test_passive_aggressive import random_state

#from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
data = pd.read_csv("test-score.csv", header = None, names=['1st','2nd','3rd','final'])



print(data.head())
print(data.shape)
print(data.isnull().sum().sum())
print(data.describe())

X=data[['1st','2nd','3rd']]
y=data[['final']]

print(X,y)


X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.7,random_state=20)
ls = LinearRegression().fit(X_train,y_train)


print(ls.coef_)
print(ls.intercept_)

print("train score: {}".format(ls.score(X_train,y_train)))
print("test score: {}".format(ls.score(X_test,y_test)))


import matplotlib.pylab as plt
plt.plot(X_train,y_train,'o')
plt.legend(['1st','2nd','3rd'])

plt.show()

