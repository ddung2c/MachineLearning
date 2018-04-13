from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import mglearn

X,y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=20)

mlp = MLPClassifier(solver='lbfgs',random_state=0)
mlp.fit(X_train,y_train)

print(mlp.score(X_test,y_test))

#########################################
mlp = MLPClassifier(solver='lbfgs',random_state=0,hidden_layer_sizes=[10,10])

mlp.fit(X_train,y_train)
print(mlp.score(X_test,y_test))

mglearn.plots.plot_2d_separator(mlp,X_train,fill=True,alpha=.3)
mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.show()
