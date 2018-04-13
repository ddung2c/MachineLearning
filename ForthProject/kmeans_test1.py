from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster.k_means_ import k_means

X,y = make_blobs(n_samples=300,centers = 4, cluster_std =0.6, random_state=0)
#plt.scatter(X[:,0],X[:,1],s=50)
#plt.show()


kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

y_kmeans=kmeans.predict(X)

print(X)
print(y_kmeans)

plt.scatter(X[:,0],X[:,1],c=y_kmeans,s=50)
centers=kmeans.cluster_centers_
plt.scatter(centers[:,0],centers[:,1],c='red',s=200,alpha=0.5)
plt.show()