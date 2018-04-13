from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=0,n_samples=30)
dbscan = DBSCAN(eps=1.0, min_samples = 2)
clusters = dbscan.fit_predict(X)
print("cluster Label:\n{}".format(clusters))


print(dbscan.eps)
print(dbscan.min_samples)
