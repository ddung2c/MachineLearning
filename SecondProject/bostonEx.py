from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

##def load_extended_boston():
#    boston = load_boston()
#    X = MinMaxScaler().fit_transform(boston.data)
#    X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    
#    return X, boston.target
boston = load_boston()
print(boston.data.shape)
print(boston.target.shape)

X = boston.data
y = boston.target

X = MinMaxScaler().fit_transform(X)
X_poly = PolynomialFeatures(degree=2, include_bias=False)
X = X_poly.fit_transform(X)
print(X.shape)
print(X_poly.get_feature_names())