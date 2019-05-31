import pandas as pd
from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf
import numpy as np


names = ['ORLY', 'DE', 'SHW', 'LMT', 'NVDA', 'NOC', 'NFLX', 'GWW', 'GOOGL', 'GOOG']

def read_stock(name):
    path = 'top_returns/' + name + '.csv'
    df = pd.read_csv(path)
    df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index)
    return df


def append_returns(TIKR):
    # TIKR is a pandas df
    df = TIKR.dropna()  # remove nans

    returns_series = df['close'] - df['open']  # series of returns
    returns_series.name = 'returns'

    pct_returns = (df['close'] - df['open']) / df['open']
    pct_returns.name = 'pct_returns'

    return pd.concat([df, returns_series, pct_returns], axis=1)


#look at auto correlation over a short time period
def short_term_acf(ts,limit = 20000, window = 120, nlags = 4, threshold = 0.2):
    #look at about 1/2 of the data
    ts = ts.head(limit)
    #ts is a pandas df
    n = len(ts)

    # create list to hold acf values
    acf_arr = []

    #get auto_correlation function values using acf from statsmodels
    for i in range(n - window - 1):
        vals = acf(ts.iloc[i:i + window])[1:nlags]

        #crude filter: remove values unless they exceed threshold
        for i in range(len(vals)):
            if abs(vals[i]) < threshold:
                vals[i] = 0
        # append to acf_arr
        acf_arr.append(vals)

    acf_arr = np.array(acf_arr)
    m = acf_arr.shape[0]

    #create new dataframe to hold values
    new_df = pd.DataFrame(ts.index[:m])

    for i in range(1,nlags):
        col_name = 'lag' + str(i)
        new_df[col_name] = acf_arr[:, i-1]
    new_df = new_df.set_index('timestamp')

    return new_df

def main():
    ticker = 'NVDA'

    stock_df = read_stock(ticker)
    stock_df = append_returns(stock_df)

    #built in autocorrelation
    '''
    acp = autocorrelation_plot(stock_df[['returns']])
    acp.set_xlim(left = 0,right=20)
    plt.show()
    '''

    acp_df = short_term_acf(stock_df[['returns']], threshold = 0.2, nlags = 3 )
    acp_df.plot()
    plt.savefig('acp_for_'+ticker)

    # remove rows where autocorrelation is not strong
    acp_df_trimm = acp_df[(acp_df != 0).all(1)]

    #spaced out every 2 hours
    #acp_df_trimm.asfreq('H')

    path = 'acp_data_for'+ticker+'.txt'
    print(acp_df_trimm.head(10))
    print(acp_df_trimm.tail(5))

if __name__ == '__main__':
    print('starting...')
    main()