from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from matplotlib.pylab import plt

X,y = make_moons(200,noise=.05, random_state=0)

print(X,y)

labels = KMeans(2,random_state=0).fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=labels,s=50, cmap='viridis')
plt.show()


