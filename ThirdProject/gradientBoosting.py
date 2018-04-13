from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,random_state=0)

#default depth:3, tree:100, learning rate =0.1
gbrt = GradientBoostingClassifier(random_state= 0)
gbrt.fit(X_train,y_train)
print("1.train accuracy: {:.3f}".format(gbrt.score(X_train,y_train)))
print("1.test accuracy:{:.3f}".format(gbrt.score(X_test,y_test)))


gbrt=GradientBoostingClassifier(random_state=0,max_depth=1)
gbrt.fit(X_train,y_train)
print("2.train accuracy: {:.3f}".format(gbrt.score(X_train,y_train)))
print("2.test accuracy: {:.3f}".format(gbrt.score(X_test,y_test)))


gbrt=GradientBoostingClassifier(random_state=0,learning_rate=0.01)
gbrt.fit(X_train,y_train)
print("3.train_accuracy:{:.3f}".format(gbrt.score(X_train,y_train)))
print("3.test accuracy: {:.3f}".format(gbrt.score(X_test,y_test)))