from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(random_state=0,n_samples=12)
linkage_array=ward(X)
dendrogram(linkage_array)
plt.show()

