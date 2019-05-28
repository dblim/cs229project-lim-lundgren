from utils import preprocess, lookback_kernel, quadratic_kernel, y_numeric_to_vector
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

# Data source
path = 'data/top_stocks/AAP.csv'
data = pd.read_csv(path)
print(data.dtypes)


# Function to extract stuff from pandas read of kaggle dataset
def kaggle_preprocess(data):
    """A preprocess function similar to that in util file. Should be moved there."""
    n, d = np.shape(data)
    open_price = np.array(data['open'])[0:n-1]
    close_price = np.array(data['close'])[0:n-1]
    high_price = np.array(data['high'])[0:n-1]
    low_price = np.array(data['low'])[0:n-1]
    traded_volume = np.array(data['volume'])[1:n] - np.array(data['volume'])[0:n - 1]
    returns = np.array(data['close']) / np.array(data['open']) - 1
    returns = returns[1:n]
    return np.column_stack((open_price, high_price, low_price, close_price, traded_volume)), returns

x_train : np.ndarray
y_train : np.ndarray

# Process data
X,Y = kaggle_preprocess(data)

# We use a 80/20 split of training/validation
training_width = int(0.80 * len(X))

x_train = X[0:training_width]
y_train = Y[0:training_width]
x_val = X[training_width:X.shape[0]]
y_val = Y[training_width:Y.shape[0]]
y_train = y_train.reshape(y_train.shape[0], 1)
y_val = y_val.reshape(y_val.shape[0], 1)

# Train the model
model = VAR(endog=x_train)
results = model.fit()

print(results)


