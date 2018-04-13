from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X,y = make_moons(n_samples=100, noise=0.25, random_state=3)

X_train, X_test, y_train,y_test = train_test_split(X,y,stratify=y, random_state=42)

forest = RandomForestClassifier(n_estimators = 5, random_state=2)
forest.fit(X_train,y_train)

print("RandomForest Train Accuracy {}".format(forest.score(X_train,y_train)))
print("RandomForest Test Accuracy {}".format(forest.score(X_test,y_test)))

import matplotlib.pyplot as plt
#plt.boxplot(x)
#plt.plot(x_train,y_train,'o')
plt.scatter(X[:,0],X[:,1],c=y)
 #X's 0th value in X axis, X's 1st value in Y axis,
plt.show()

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train,y_train)

print("DecisionTree train accuracy: {}".format(tree.score(X_train,y_train)))
print("DecisionTree test accuracy: {}".format(tree.score(X_test,y_test)))