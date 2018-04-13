from sklearn.datasets import load_breast_cancer
import numpy as np
#from sklearn.linear_model.tests.test_passive_aggressive import random_state

cancer = load_breast_cancer()

#print(cancer.keys())

#print(cancer.data.shape)

#print(cancer.target)
#print(cancer.feature_names)


#print("count per class:\n{}".format({n:v for n,v in zip(cancer.target_names,np.bincount(cancer.target))}))


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target, random_state=66)

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pylab as plt #그래프 그리기


training_accuarcy=[]
test_accuracy=[]
neighbors_settings=range(1,11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train,y_train)
    training_accuarcy.append(clf.score(X_train,y_train))
    test_accuracy.append(clf.score(X_test,y_test))
    
plt.plot(neighbors_settings,training_accuarcy, label = "train accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

print(clf.score(X_test,y_test))


