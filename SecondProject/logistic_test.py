from sklearn.datasets import load_digits

digit = load_digits()

print(digit.data.shape)  

print(digit.data)
print(digit.target.shape)
print(digit.target)


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(20,4))

for index, (image, label) in enumerate(zip(digit.data[0:5],digit.target[0:5])):
    plt.subplot(1,5,index+1) 
    plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
plt.show()
    
    
