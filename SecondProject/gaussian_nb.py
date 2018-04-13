from sklearn.datasets import load_iris

iris = load_iris()

#print(iris) #setal width,length / fatal width, length


#print(iris.data)
#print(iris.target)

X=iris.data
y=iris.target

print(X.shape,y.shape) #(150,4) (150,) 

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=50)

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train,y_train)



print("predict:       {}".format(model.predict(X_test[0:20])))
print("answer:        {}".format(y_test[0:20]))


print(model.score(X_train,y_train))
print(model.score(X_test,y_test))
