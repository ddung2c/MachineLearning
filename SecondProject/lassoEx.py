from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
import numpy as np

def load_extended_boston():
    boston = load_boston()
    X = MinMaxScaler().fit_transform(boston.data)
    X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    
    return X, boston.target

boston = load_boston()

X, y= load_extended_boston()
#X = boston.data
#y = boston.target

from sklearn.linear_model import Lasso
X_train,X_test, y_train,y_test = train_test_split(X,y,random_state=0)

lasso = Lasso().fit(X_train,y_train)
print("lasso train accuracy: {:.2f}".format(lasso.score(X_train,y_train)))
print("lasso test accuracy: {:.2f}".format(lasso.score(X_test,y_test)))

lasso10 = Lasso(alpha=10).fit(X_train,y_train)
print("lasso10 train accuracy: {:.2f}".format(lasso10.score(X_train,y_train)))
print("lasso10 test accuracy: {:.2f}".format(lasso10.score(X_test,y_test)))

lasso01 = Lasso(alpha=0.1).fit(X_train,y_train)
print("lasso0.1 train accuracy: {:.2f}".format(lasso01.score(X_train,y_train)))
print("lasso0.1 test accuracy: {:.2f}".format(lasso01.score(X_test,y_test)))

lasso001 = Lasso(alpha=0.001).fit(X_train,y_train)
print("lasso0.001 train accuracy: {:.2f}".format(lasso001.score(X_train,y_train)))
print("lasso0.001 test accuracy: {:.2f}".format(lasso001.score(X_test,y_test)))
