from var_utils import combine_ts_returns, minutizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

"""In this script, we will get the training residuals for VAR. These will then be trained on using RNN/LSTM"""

justin_data : bool=True

# data
if justin_data is True:
    path = '../data/sectors/Information Technology/'
    tickers = ['ACN', 'AMAT', 'CDNS', 'IBM', 'INTU', 'LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']

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
results = VAR(endog_y).fit(1)
#print(results.params)


# Predictions on test

predictions_test = np.zeros(( test_data.shape[0] ,test_data.shape[1]    ))

for i in range(0,test_num_steps):

    predictions_test[i] = results.forecast(test_data.values[i,:].reshape(1,test_data.shape[1]),steps = 1)
print(predictions_test)

# Save predictions
pd.DataFrame(predictions_test).to_csv('../output/VAR_results/test_predictions.csv',
                                       index=False)


# Hyperparameter tuning
# We get an optimal hyperparameter of p=1
# This is probably wrong
def var_mse_p(endog_y, p):
    results = VAR(endog_y).fit(p)
    predictions = results.forecast(endog_y.values, steps=val_num_steps)
    pd.DataFrame(predictions).to_csv('../output/VAR_results/' + str(p) + '_train_prediction.csv')
    MSE = np.sum(predictions - endog_y_val.values, axis=0) ** 2 / val_num_steps
    MSE_average = np.sum(MSE) / num_stocks
    return (MSE_average)


# We now get the smallest MSE over all p in range (1, max_p)
def optimal_p(endog_y, max_p):
    p_list = ['{}'.format(p) for p in range(1, max_p)]
    MSE_list = [var_mse_p(endog_y, p) for p in range(1, max_p)]
    MSE_dictionary = dict(zip(p_list, MSE_list))
    return min(MSE_dictionary, key=MSE_dictionary.get)


# Try max_p = 10, maximum hyperparameter to search. We get p = 1.
def hyperparameter_search(endog_y, max_p):
    return (optimal_p(endog_y, max_p))


