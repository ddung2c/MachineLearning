import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


data = pd.read_csv("diabetes.csv", header = None, names=['1st','2nd','3rd','4th','5th','6th','7th','8th','final'])

print("data.head")
print(data.head())
print("data.shape")
print(data.shape)
print("data.isnull().sum().sum()") 
print(data.isnull().sum().sum())  

print("data.describe()")
print(data.describe())

print(data['final'].head())

X=data[['1st','2nd','3rd','4th','5th','6th','7th','8th']]
y=data['final']

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.7,random_state=20)
ls = LogisticRegression()
ls.fit(X_train,y_train)


print(ls.predict(X_test[0:10])) #predict value
predictions = ls.predict(X_test) #answer
print(y_test[0:10])

score = ls.score(X_test,y_test)
print(score)