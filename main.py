import numpy as np

from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt


test_case = np.array([2,-2.5])
X,Y = make_blobs(n_samples = 400,n_features = 2,centers = 2)
plt.style.use('classic')


plt.scatter(X[:,0],X[:,1],c=Y)
plt.scatter(test_case[0],test_case[1],color = 'green',marker = '*')





def distance(a1,a2):
    return np.sum((a1-a2)**2)**0.5


# a1 = np.array([2,3])
# a2 = np.array([4,5])

# distance(a1,a2)


def KNN(X,Y,test_point,j = 5):
    
   
    m = X.shape[0]
    values = []

    for i in range(m):
        
        d = distance(test_point,X[i])
        values.append((d,Y[i]))
        
    values = sorted(values)
    values = np.array(values[:j]) # for slicing we need a np array
    
    values = values[:,1] # we only need the labels
    b = np.unique(values,return_counts= True)
    
    idx = np.argmax(b[1]) #index of max arguement
    pred = b[0][idx]  #selecting max arguement
    
    return int(pred)
    
KNN(X,Y,test_case)
import pandas as pd

df = pd.read_csv("mnist_train.csv")






df.shape

data = df.values   # converts dataframe into np array

Y = data[:,0]    # value ie numbers
X = data[:,1:]   # corresponding pixel values

# we have our data



plt.imshow(X[0].reshape(28,28),cmap='gray')

# corresponding label
Y[0]

# for i in range(10):
#     plt.imshow(X[i].reshape(28,28),cmap='gray')
#     plt.show()  # why plt.show() used to show all if not only Y[9] shown        
    

# to make  them shown in a grid of 2*5


plt.figure()
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X[i].reshape(28,28),cmap='gray')
    plt.title(str(Y[i]))
    plt.axis('off')
    
#outside as first plot images on graph then display the whole graph    
plt.show()    

test_sample = X[6] #actuallty fourth image

pred = KNN(X,Y,test_sample)

print(pred)

actual_value = Y[6]

print(actual_value)

# now run on test cases