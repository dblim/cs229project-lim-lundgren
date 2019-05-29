import pandas as pd
from pandas import Series
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR

# A function to combine tickers and add a column for returns.
# Returns defined as closing minus opening price.


def append_returns(stock, stock_name):
    df = stock.dropna() # remove nans
    returns = stock['close'] - stock['open']
    returns.name = stock_name + '_returns'
    return pd.concat([df, returns], axis = 1)

def combine_ts_returns(tickers : list ):
    stock0 = tickers[0]
    data = pd.read_csv('data/top_stocks/'+stock0+'.csv')
    data = append_returns(data, stock0)
    renamer = {'close': stock0 + '_close', 'high': stock0 + '_high', 'low': stock0 + '_low',
               'open': stock0 + '_open', 'volume': stock0 + '_volume', }
    data = data.rename(columns = renamer )
    tickers.remove(stock0)
    return(data)

tickers = ['AAP']
print(combine_ts_returns(tickers))
print(tickers )










# Split data
#n = np.shape(data)[0]
#train_data = data[0 : int(0.8*n)]
#vald_data = data[int(0.8*n) : ]



