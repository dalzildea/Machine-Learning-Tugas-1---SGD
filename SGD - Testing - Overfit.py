
# coding: utf-8

# In[13]:


import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import argparse
from pydataset import data

def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))

def next_batch(X, y, batchSize):
    for i in np.arange(0, X.shape[0], batchSize):
        yield (X[i:i + batchSize], y[i:i + batchSize])

alpha = 0.1
epochs = 100
batchSize = 100
dataset = data('iris')

(X, y) = data.iris(index=list(range(0,20,1)), columns=list(range(0,3,1)))

X = np.c_[np.ones((X.shape[0])), X]
 
W = np.random.uniform(size=(X.shape[1],))
 
lossHistory = []

for epoch in np.arange(0, epochs):
    epochLoss = []
 
    for (batchX, batchY) in next_batch(X, y, batchSize):
        preds = sigmoid_activation(batchX.dot(W))
 
        error = preds - batchY
 
        loss = np.sum(error ** 2)
        epochLoss.append(loss)
 
        gradient = batchX.T.dot(error) / batchX.shape[0]
 
        W += alpha * gradient
 
    lossHistory.append(np.average(epochLoss))
    
Y = (-W[0] - (W[1] * X)) / W[2]

fig = plt.figure()
plt.plot(np.arange(0, epochs), lossHistory)
plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
fig.subtitle("Error Rate")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((0, 1))
plt.ylim((-2, 2))
plt.legend(loc="best")
plt.show()

