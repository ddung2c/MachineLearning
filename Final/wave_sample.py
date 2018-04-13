import numpy as np
import mglearn

X,y = mglearn.datasets.make_wave(n_samples=100)
bins = np.linspace(-3,3,11)


print(bins)
 

which_bin = np.digitize(X, bins=bins)

print(which_bin[:5])
print(X[:5])

print("\ndata:\n", X[:5])
print("\nwhich_bin:\n", which_bin[:5])


print(np.min(X),np.max(X))