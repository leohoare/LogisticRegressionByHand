
import pandas as pd
import numpy as np

def logistic_regression(data, labels, weights, num_epochs, learning_rate): # do not change the heading of the function
    for epoch in range(num_epochs):
        z = weights[0] + np.dot(data, weights[1:])
        sig = 1 / (1 + np.exp(-z))
        loss = sig - labels
        dotprod = data.T.dot(loss)
        weights[0] = weights[0]-learning_rate*loss.sum()
        weights[1:] = weights[1:]-learning_rate*dotprod
        if epoch%100 == 0:
            print(weights)
    return weights

# data_file='./asset/a'
# raw_data = pd.read_csv(data_file, sep=',')
# labels=raw_data['Label'].values
# data=np.stack((raw_data['Col1'].values,raw_data['Col2'].values), axis=-1)
# ## Fixed Parameters. Please do not change values of these parameters...
# weights = np.zeros(3) # We compute the weight for the intercept as well...
# num_epochs = 50000
# learning_rate = 50e-5
# coefficients = logistic_regression(data, labels, weights, num_epochs, learning_rate)