import pandas as pd
import numpy as np

'''
Logistic Regression Function using gradient descent
data - X matrix, (n,i) where n is rows, i is number of variables
labels - Y matrix (n,1)
num_epochs - desired epochs
learning rate - rate at which model learns
'''
def logistic_regression(data, labels, weights, num_epochs, learning_rate): 
    for epoch in range(num_epochs):
        z = weights[0] + np.dot(data, weights[1:])
        sig = 1 / (1 + np.exp(-z))
        loss = sig - labels
        dotprod = data.T.dot(loss)
        weights[0] = weights[0]-learning_rate*loss.sum()
        weights[1:] = weights[1:]-learning_rate*dotprod
        if epoch%100 == 0:
            print(f'epoch:{epoch} weights:{weights}')
    return weights

data_file ='./data'
raw_data = pd.read_csv(data_file, sep=',')
labels = raw_data['Label'].values
data = raw_data[['Col1','Col2']]
weights = np.zeros(data.shape[1] + 1) # We compute the weight for the intercept as well...
num_epochs = 20000
learning_rate = 50e-5
coefficients = logistic_regression(data, labels, weights, num_epochs, learning_rate)
print("Training Complete")
print(f"Coefficients: {coefficients}")
