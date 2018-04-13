import pandas as pd
import matplotlib.pylab as plt
from sklearn.tree.tree import DecisionTreeRegressor

ram_prices = pd.read_csv("ram_price.csv")

print(ram_prices)

plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel("year")
plt.ylabel("price ($/Mbyte)")
plt.show()

#############################################

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import numpy as np

data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >=2000]

X_train = data_train.date[:,np.newaxis]
y_train = np.log(data_train.price)

tree = DecisionTreeRegressor().fit(X_train,y_train)
linear_reg = LinearRegression().fit(X_train,y_train) 

X_all = ram_prices.date[:,np.newaxis]

pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

price_tree = np.exp(pred_tree)
price_lr=np.exp(pred_lr)

plt.semilogy(data_train.date,data_train.price, label = "train data")
plt.semilogy(data_test.date,data_test.price, label = "test data")
plt.semilogy(ram_prices.date, price_tree, label = "Tree prediction")
plt.semilogy(ram_prices.date,price_lr,label = "Linear regression")
plt.legend()
plt.show()