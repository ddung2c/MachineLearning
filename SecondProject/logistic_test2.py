from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
#from numpy.polynomial.tests.test_classes import random
digit = load_digits()

print(digit.data.shape)  
##(1797, 64) 8*8 -> 1 line as 64

print(digit.data)
print(digit.target.shape)
print(digit.target)


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(20,4))

for index, (image, label) in enumerate(zip(digit.data[0:5],digit.target[0:5])):
    plt.subplot(1,5,index+1) #1 column 5 raw, in this location
    plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
plt.show()
    
    
X_train,X_test,y_train,y_test = train_test_split(digit.data,digit.target
                                                 ,random_state=0,test_size=.25)


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(verbose=1)
lr.fit(X_train,y_train)

print(lr.score(X_train,y_train))
print(lr.score(X_test,y_test))

print(lr.predict(X_test[0].reshape(1,-1))) #predict value
print(y_test[0]) #answer

print(lr.predict(X_test[9].reshape(1,-1))) #2 dimention, rank is less than 1 so reshape as 1,-1
print(y_test[9]) #answer


