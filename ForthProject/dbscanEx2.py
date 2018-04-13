from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from skleanr.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

scaler = StandardScaler()
scaler.fit(BaseException