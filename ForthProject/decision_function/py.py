from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris, make_circles
from sklearn.model_selection import train_test_split
import numpy as np

X,Y= make_circles(noise=0.25, factor=0.5, random_state=1)

#change class name
y_named=np.array(["blue","red"])[Y]

X_train,X_test,y_train_named,y_test_named,y_train,y_test =\
train_test_split(X,y_named,Y,random_state=0)

gbrt=GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train,y_train_named)

print("X_test.shape:{}".format(X_test.shape))
print("Result of Decision Function:{}".format(gbrt.decision_function(X_test).shape))

print("Decision Function:\n{}".format(gbrt.decision_function(X_test)[:6]))
print("Threshold vs. Decision Tree:\n{}".format(gbrt.decision_function(X_test)>0))
print("Predict:\n{}".format(gbrt.predict(X_test)))

greater_zero = (gbrt.decision_function(X_test)>0).astype(int)
pred = gbrt.classes_[greater_zero]
print("pred is same as predictions:\n{}".format(np.all(pred==gbrt.predict(X_test))))

decision_function=gbrt.decision_function(X_test)
print("Decision Function Min:\n{:.2f} Max:{:.2f}".format(np.min(decision_function),np.max(decision_function)))

print("Predicted probabilities:\n{}".format(gbrt.predict_proba(X_test[:6])))

