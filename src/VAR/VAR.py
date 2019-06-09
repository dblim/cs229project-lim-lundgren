from var_utils import combine_ts, minutizer, preprocess_2_multi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

"""In this script, we will get the training residuals for VAR. These will then be trained on using RNN/LSTM"""
transpose : bool = False
justin_data : bool=True

# data
if justin_data is True:
    path = '../data/sectors/Information Technology/'
    tickers = ['ACN', 'AMAT']# 'CDNS', 'IBM', 'INTU', 'LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']

if justin_data is False:
    path = '../data/top_stocks/'
    tickers = ['AAP', 'AES']

data = pd.read_csv('../data/preprocessed_time_series_data.csv')
data = data.drop(columns=['Unnamed: 0'])


new_data = np.zeros((data.shape[0], 10))
for i in range(10):
    new_data[:, i] = data.values[:, 4*i]

data = pd.DataFrame(new_data)



# Split data
n = np.shape(data)[0]
train_data = data[0 : int(0.7*n)]
val_data = data[int(0.7*n) : int(0.85*n)]
test_data = data[ int(0.85*n) : ]

# Train on returns
endog_y = train_data

# Validate
endog_y_val = val_data

# Test
endog_y_test = test_data

# Validation start and end dates.
val_start_index = val_data.index.min()
val_end_index = val_data.index.max()

# Hyperparameter searching

# The function below returns the average MSE given a batch of stocks for a particular value of p
# Recall when we do predictions, we should predict over the length of time that is the validation set

val_num_steps = len(endog_y_val)
test_num_steps = len(endog_y_test)
num_stocks = len(tickers)

#Model
model = VAR(endog_y)
results = model.fit(1)


# Convert params to np.array

params = results.params.values

# This stuff here because I don't understand if statsmodels API returns the transpose of the parameters that we want
if transpose is True:
    bias = params[0,:].reshape(1,10)

    weights = np.delete(params, (0), axis = 0)
    weights = weights.T
    params = np.r_[bias, weights ]



# Convert test data to numpy array and at first column of bias
bias_vector = np.ones(len(test_data)).reshape(len(test_data),1)

test_data = test_data.values
test_data = np.c_[bias_vector, test_data]
# print(test_data[:,0]) - This here to check we have really added bias.

# Prediction. Note we always start at 1 time step into test set rather than at the 0th.
test_predictions = np.zeros(( len(test_data), 10))

for i in range(1, len(test_data)):
    test_predictions[i,:] =np.dot( test_data[i,: ].reshape(1,11), params)

# Compare to LSTM. Detele first 24 columns
test_predictions = test_predictions[ 24: , : ]
print(test_predictions)

# Save to pandas dataframe. This is test predictions 24 time steps into LSTM.
df = pd.DataFrame(test_predictions)
df.to_csv('../output/VAR_results/VAR_test_predictions.csv')


