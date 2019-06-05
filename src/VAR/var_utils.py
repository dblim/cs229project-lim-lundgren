import pandas as pd

# Appends returns column to dataframe

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
