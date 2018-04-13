from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

boston = load_boston()
print(boston.data.shape)
print(boston.target.shape)

x = boston.data
y = boston.target

x_train,x_test, y_train,y_test = train_test_split(x,y,random_state=50)
lr = LinearRegression().fit(x_train,y_train)

print(lr.coef_)   
print(lr.intercept_) 

print("train score:{}".format(lr.score(x_train,y_train)))

import matplotlib.pylab as plt
plt.boxplot(x)
#plt.plot(x_train,y_train,'o')
plt.show()


#import numpy as np
#print(np.max(lr.coef_))  max값 출력
#print(np.argmax(lr.coef_)) #몇번쨰 index가 가장 max 값인지

