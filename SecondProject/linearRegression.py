from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mglearn
#from sklearn.linear_model.tests.test_passive_aggressive import random_state

X, y = mglearn.datasets.make_wave(n_samples=1000)

print(X.shape,y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)

lr = LinearRegression().fit(X_train, y_train)

print(lr.coef_)   #[0.39390555]
print(lr.intercept_)   #-0.03180434302675973
##y=0.39x-0.03

#accuracy
print("train score:{}".format(lr.score(X_train,y_train)))
print("test score:{}".format(lr.score(X_test,y_test)))


import matplotlib.pylab as plt
plt.plot(X_train,y_train,'o',X,lr.predict(X))
plt.plot(X_test,y_test,'o',X,lr.predict(X))
plt.show()