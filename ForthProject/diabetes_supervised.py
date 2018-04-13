from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

dataset = pd.read_csv("diabetes.csv",names=[1,2,3,4,5,6,7,8,9],header=None)
print(dataset.head())
print(dataset.shape)
print(dataset.isnull().sum().sum())
print(dataset.describe())


X = dataset[[1,2,3,4,5,6,7,8]]
y= dataset[9]
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y, test_size=.25)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

#Logistic
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(verbose=1)
lr.fit(X_train,y_train)
print("\n")
print("logistic:\t{}".format(lr.score(X_train,y_train)))
print("logistic:\t{}".format(lr.score(X_test,y_test)))


#KNN
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)
print("KNN:\t\t{}".format(clf.score(X_train,y_train)))
print("KNN:\t\t{}".format(clf.score(X_test,y_test)))

#Tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train,y_train)
print("train accuracy:\t{:.3f}".format(tree.score(X_train,y_train)))
print("test accuracy:\t{:.3f}".format(tree.score(X_test,y_test)))

#Forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 5, random_state=2)
forest.fit(X_train,y_train)
print("RandomForest:\t{}".format(forest.score(X_train,y_train)))
print("RandomForest:\t{}".format(forest.score(X_test,y_test)))

#SVM
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)
print("SVM:\t\t{}".format(svc.score(X_train,y_train)))
print("SVM:\t\t{}".format(svc.score(X_test,y_test)))
print("predict:       {}".format(svc.predict(X_test)[0:20]))
print("answer:        {}".format(y_test[0:20]))
#MLP
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs',random_state=0)
mlp.fit(X_train,y_train)
print("MLP:\t\t{}".format(mlp.score(X_train,y_train)))
print("MLP:\t\t{}".format(mlp.score(X_test,y_test)))
print("predict:       {}".format(mlp.predict(X_test)[0:20]))
print("answer:        {}".format(y_test[0:20].values.reshape(1,-1)))


import matplotlib.pylab as plt
plt.boxplot(X)
#plt.plot(x_train,y_train,'o')
plt.show()