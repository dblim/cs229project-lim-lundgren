from var_utils import combine_ts_returns
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

data = combine_ts_returns(path, tickers)

# Split data
n = np.shape(data)[0]
train_data = data[0 : int(0.7*n)]
val_data = data[int(0.7*n) : int(0.85*n)]
test_data = data[ int(0.85*n) : ]

# Train on returns
y_list = [t+'_returns' for t in tickers]
endog_y = train_data[y_list]
exog_x = train_data.drop(columns=y_list)

# Validate
endog_y_val = val_data[y_list]
exog_x_val = val_data.drop(columns=y_list)

# Test
endog_y_test = test_data[y_list]
exog_x_test = test_data.drop(columns = y_list)

# Validation start and end dates.
val_start_index = val_data.index.min()
val_end_index = val_data.index.max()

# Hyperparameter searching

# The function below returns the average MSE given a batch of stocks for a particular value of p
# Recall when we do predictions, we should predict over the length of time that is the validation set

def var_mse_p(endog_y, p):
        val_num_steps = len(endog_y_val)
        num_stocks = len(tickers)
        results = VAR(endog_y).fit(p)
        predictions = results.forecast(endog_y.values, steps = val_num_steps)
        pd.DataFrame(predictions).to_csv('../output/VAR_results/' +str(p)+ '_train_prediction.csv')
        MSE = np.sum(predictions - endog_y_val.values, axis = 0)**2/val_num_steps
        MSE_average = np.sum(MSE)/num_stocks
        return (MSE_average)

# We now get the smallest MSE over all p in range (1, max_p)
def optimal_p(endog_y,max_p):
    p_list = ['{}'.format(p) for p in range(1,max_p)]
    MSE_list = [var_mse_p(endog_y,p) for p in range(1, max_p)]
    MSE_dictionary = dict(zip(p_list, MSE_list))
    return min(MSE_dictionary, key=MSE_dictionary.get)

# Try max_p = 10, maximum hyperparameter to search

def hyperparameter_search(endog_y, max_p):
    return(optimal_p(endog_y,max_p))

print(hyperparameter_search(endog_y,10))

# AIC selection
#for p in range(1,10):
#    p += 1
#    model =  VAR(endog_y).fit(p)
#    print( model.aic)


# We get an optimal hyperparameter of p=1


