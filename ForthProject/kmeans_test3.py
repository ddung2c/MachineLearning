import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris=load_iris()
X, y = iris.data, iris.target

#print(X,y)

kmeans=KMeans(n_clusters=3)
kmeans.fit(X)

y_kmeans = kmeans.predict(X)

plt.scatter(X[:,0],X[:,1],c=y_kmeans,s=50)
centers=kmeans.cluster_centers_
plt.scatter(centers[:,0],centers[:,1],c='red',s=200,alpha=0.5)
plt.show()

print(y_kmeans)
print(y)

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y,y_kmeans)
print(mat)