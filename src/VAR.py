import pandas as pd
from pandas import Series
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

# A function that appends an extra column for returns at the end of a stock data frame

def append_returns(stock, stock_name):
    df = stock.dropna() # remove nans
    returns = stock['close'] - stock['open']
    returns.name = stock_name + '_returns'
    return pd.concat([df, returns], axis = 1)

    """A function that takes a list of tickers  and produces a table with columns indexed as:
        close, high, low, open volume and returns. So for example, if tickers = ['AAP', 'AES'], then
        the function below will produce a pandas dataframe of columns indexed as 
        ['AAP_close' 'AAP_high' 'AAP_low' 'AAP_open' 'AAP_volume' 'AAP_returns'
         'AES_close' 'AES_high' 'AES_low' 'AES_open' 'AES_volume' 'AES_returns']"""

def combine_ts_returns(tickers : list ):
    stock0 = tickers[0]
    data = pd.read_csv('data/top_stocks/'+stock0+'.csv', index_col = "timestamp", parse_dates = True)
    data = append_returns(data, stock0)
    renamer = {'close': stock0 + '_close', 'high': stock0 + '_high', 'low': stock0 + '_low',
               'open': stock0 + '_open', 'volume': stock0 + '_volume', }
    data = data.rename(columns = renamer )
    tickers.remove(stock0)
    for str in tickers:
        new_data = pd.read_csv('data/top_stocks/' + str + '.csv', index_col="timestamp", parse_dates=True)
        new_data = append_returns(new_data, str)
        renamer = {'close': str + '_close', 'high': str + '_high', 'low': str + '_low',
                   'open': str + '_open', 'volume': str + '_volume', }
        new_data = new_data.rename(columns=renamer)
        data = pd.concat([data, new_data], axis=1, sort=True)
    tickers.append(stock0)
    return data.interpolate()[1 : data.shape[0]]


# data
tickers = ['AAP', 'AES', 'AMD', 'BSX']


data = combine_ts_returns(tickers)



# Split data
n = np.shape(data)[0]
train_data = data[0 : int(0.8*n)]
val_data = data[int(0.8*n) : ]

# Train on returns
y_list = [t+'_returns' for t in tickers]
endog_y = train_data[y_list]
exog_x = train_data.drop(columns=y_list)

# Validate
endog_y_val = val_data[y_list]
exog_x_val = val_data.drop(columns=y_list)

# For each p value,  print MSE error corresponding to VAR model.
for p in range(1,10):
    model = VAR(exog_x.values)
    predictions = model.fit(p).forecast( exog_x_val.values, steps=1)
    for i, ticker in  enumerate( tickers):
        MSE = sum((predictions[:, i] - endog_y_val.values[:, i])**2)/endog_y_val.shape[0]
        print("For p = {} and {}, ".format(p, ticker) + "MSE error is", MSE)

