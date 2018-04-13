from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X)

cm2 = ListedColormap(['#0000aa','#ff2020'])
plt.scatter(X[:,0],X[:,1],c=clusters,cmap=cm2,s=60,edgecolors='black')
plt.xlabel("feature 0")
plt.ylabel("feature 1")

plt.show()