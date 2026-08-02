from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
#==============================neural network from scratch===============================================
def ReLu(Z):
    res = np.zeros(len(Z))
    for i in range(len(res)):
        res[i] = max(Z[i],0)
    return res 
  
def soft_max(z):
    res = np.zeros(len(z))
    for i in range(len(res)):
        res[i] = np.exp(z[i])/sum(np.exp(z[j]) for j in range(len(z)))
    return res    

def dérivée_ReLU(z):
    res = np.zeros(len(z))
    for i in range(len(z)):
        if z[i] < 0 :
            res[i] = 0
        else:
            res[i] = 1
    return res            

def pred(X,W1,W2,b1,b2):
    Z1 = X@W1 + b1
    a1 = ReLu(Z1)
    Z2 = a1@W2 + b2
    prediction = soft_max(Z2)
    return prediction


digits = load_digits()

images = digits.data      
number = digits.target    

images,images_test,number,number_test = train_test_split(
    digits.data / 16.0, digits.target, test_size=0.2, random_state=42
)

b1 = np.zeros(16)
b2 = np.zeros(10)

W1 = np.random.randn(64, 16) * np.sqrt(2 / 64)
W2 = np.random.randn(16, 10) * np.sqrt(2 / 16)
vraie_etiquette_one_hot = np.zeros(10)
for epoch in range(100):
    for i in range(len(images)):
        X = images[i]
        Z1 = X@W1 + b1
        a1 = ReLu(Z1)
        Z2 = a1@W2 + b2
        prediction = soft_max(Z2)


        vraie_etiquette_one_hot[number[i]] = 1
        dz2 = prediction - vraie_etiquette_one_hot
        dW2 = np.outer(a1, dz2)
        db2 = dz2
        da1 = dz2 @ np.transpose(W2)
        dz1 = da1 * dérivée_ReLU(Z1)
        dW1 = np.outer(X, dz1)
        db1 = dz1

        W1 = W1 - 0.001 * dW1
        b1 = b1 - 0.001 * db1
        W2 = W2 - 0.001 * dW2
        b2 = b2 - 0.001 * db2

        vraie_etiquette_one_hot[number[i]] = 0

correct = 0 
for i in range(len(images_test)):
    L = pred(images_test[i],W1,W2,b1,b2)
    if np.where(L == L.max())[0].item() == number_test[i]:
        correct +=1

accuracy = correct/len(images_test)

L = pred((images_test[10]),W1,W2,b1,b2)
print(np.where(L == L.max())[0].item())


plt.imshow(images_test[10].reshape(8, 8), cmap='gray')
plt.title(f"Label: {number_test[10]}")
plt.show()
