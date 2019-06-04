#from utils import combine_ts_returns
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

def append_returns(stock, stock_name):
    df = stock.dropna() # remove nans
    returns = stock['close'] - stock['open']
    returns.name = stock_name + '_returns'
    return pd.concat([df, returns], axis = 1)


def combine_ts_returns(path : str, tickers : list ):
    """A function that takes a list of tickers  and produces a table with columns indexed as:
            close, high, low, open volume and returns. So for example, if tickers = ['AAP', 'AES'], then
            the function below will produce a pandas dataframe of columns indexed as
            ['AAP_close' 'AAP_high' 'AAP_low' 'AAP_open' 'AAP_volume' 'AAP_returns'
             'AES_close' 'AES_high' 'AES_low' 'AES_open' 'AES_volume' 'AES_returns']"""
    stock0 = tickers[0]
    #path = '../data/top_stocks/'
    data = pd.read_csv(path +stock0+'.csv', index_col = "timestamp", parse_dates = True)
    data = append_returns(data, stock0)
    renamer = {'close': stock0 + '_close', 'high': stock0 + '_high', 'low': stock0 + '_low',
               'open': stock0 + '_open', 'volume': stock0 + '_volume', }
    data = data.rename(columns = renamer )
    tickers.remove(stock0)
    for str in tickers:
        new_data = pd.read_csv(path + str + '.csv', index_col="timestamp", parse_dates=True)
        new_data = append_returns(new_data, str)
        renamer = {'close': str + '_close', 'high': str + '_high', 'low': str + '_low',
                   'open': str + '_open', 'volume': str + '_volume', }
        new_data = new_data.rename(columns=renamer)
        data = pd.concat([data, new_data], axis=1, sort=True)
    tickers.append(stock0)
    return data.interpolate()[1 : data.shape[0]]

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
max_p = 10
print(optimal_p(endog_y,max_p))

# AIC selection
#for p in range(1,10):
#    p += 1
#    model =  VAR(endog_y).fit(p)
#    print( model.aic)


# We get an optimal hyperparameter of p=1


