from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

def load_extended_boston():
    boston = load_boston()
    X = MinMaxScaler().fit_transform(boston.data)
    X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    
    return X, boston.target

boston = load_boston()

X, y= load_extended_boston()
#X = boston.data
#y = boston.target

from sklearn.linear_model import Ridge
X_train,X_test, y_train,y_test = train_test_split(X,y,random_state=0)

ridge = Ridge().fit(X_train,y_train)
print("train accuracy: {:.2f}".format(ridge.score(X_train,y_train)))
print("test accuracy: {:.2f}".format(ridge.score(X_test,y_test)))

redge10 = Ridge(alpha=10).fit(X_train,y_train)
print("10 train accuracy: {}".format(redge10.score(X_train,y_train)))
print("10 test accuracy: {}".format(redge10.score(X_test,y_test)))

redge01 = Ridge(alpha=0.1).fit(X_train,y_train)
print("0.1 train accuracy: {}".format(redge01.score(X_train,y_train)))
print("0.1 test accuracy: {}".format(redge01.score(X_test,y_test)))


