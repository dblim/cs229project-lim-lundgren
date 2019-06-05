import numpy as np
import pandas as pd


def preprocess(data,
               alpha_yahoo: str = 'alpha',
               incremental_data: bool = False,
               output_variable: str = 'binary',
               partitions: list = None):
    """ Function takes in an input data on the typical panda-form from fix_yahoo_finance or alpha vantage or
        any similar package. It then returns all the features as an X and the returns (binomial, multinomial or
        continuous) as a Y.

        X is an n-1 by 5 matrix with the features Open, High, Low, Close, Volume (in that particular order).

        Note that the data for X is on time unit i, whereas Y is on time unit i + 1.

        output_variable can also be any other word than "binary" or "multinomial",
        which would correspond to arithmetic returns.

        partitions needs to be active to use the multinomial case. e.g. partitions = [-0.02, -0.01, 0.0, 0.01, 0.02]
    """
    op: str
    high: str
    low: str
    close: str
    vol: str
    if alpha_yahoo == 'alpha':
        op = '1. open'
        high = '2. high'
        low = '3. low'
        close = '4. close'
        vol = '5. volume'
    elif alpha_yahoo == 'yahoo':
        op = 'open'
        high = 'high'
        low = 'low'
        close = 'close'
        vol = 'volume'
    n, _ = data.shape
    open_price = np.array(data[op])[0:n - 1]
    high_price = np.array(data[high])[0:n - 1]
    low_price = np.array(data[low])[0:n - 1]
    close_price = np.array(data[close])[0:n - 1]
    traded_volume = np.array(data[vol])[1:n] - np.array(data[vol])[0:n - 1]
    returns = np.array(data[close]) / np.array(data[op]) - 1
    returns = returns[1:n]
    if output_variable == 'binary':
        returns[returns > 0] = 1
        returns[returns < 0] = 0
    elif output_variable == 'multinomial':
        assert partitions is not None, "Missing partitions for multinomial case."
        m = len(partitions)
        for i in range(m):
            returns[returns < partitions[i]] = i + 1
        returns[returns < 1] = m + 1
        returns -= 1
    if incremental_data is True:
        time_unit_returns = close_price / open_price - 1
        high_to_open = high_price / open_price - 1
        low_to_open = low_price / open_price - 1
        return np.column_stack((time_unit_returns, high_to_open, low_to_open, traded_volume)), returns
    else:
        return np.column_stack((open_price, high_price, low_price, close_price, traded_volume)), returns



def quadratic_kernel(data):
    n, d = data.shape
    new_data = np.zeros((n, 1))
    for i in range(d):
        for j in range(d - i):
            new_data = np.hstack((new_data, (data[:, i] * data[:, j]).reshape(n, 1)))
    return np.hstack((data, new_data[:, 1:]))


def lookback_kernel(x, y,
                    periods: int = 3):
    n, d = x.shape
    y = y[periods - 1:n]
    new_data = np.zeros((n - periods + 1, 1))
    for i in range(d):
        for j in range(periods):
            new_column = (x[(periods - 1 - j):(n - j), i]).reshape(n - periods + 1, 1)
            new_data = np.hstack((new_data, new_column))
    return new_data[:, 1:], y


def y_numeric_to_vector(data, k):
    y = np.zeros((k, data.shape[0]))
    for i in range(data.shape[0]):
        y[int(data[i]), i] = 1
    return y.T


def combine_ts(tickers: list):
    stock0 = tickers[0]
    path = '../data/sectors/Information Technology/'+stock0+'.csv'
    data = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    renamer = {'close': stock0+'_close', 'high': stock0+'_high', 'low': stock0+'_low',
               'open': stock0+'_open', 'volume': stock0+'_volume', }
    data = data.rename(columns=renamer)
    tickers.remove(tickers[0])
    for str in tickers:
        path = '../data/sectors/Information Technology/'+str+'.csv'
        new_data = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        renamer = {'close': str+'_close', 'high': str+'_high', 'low': str+'_low',
                   'open': str+'_open', 'volume': str+'_volume', }
        new_data = new_data.rename(columns=renamer)

        data = pd.concat([data, new_data], axis=1, sort=True)
    tickers.insert(0, stock0)
    return data.interpolate()[1:data.shape[0]]


def minutizer(data, split: int = 5, ground_features: int = 5):
    n, d = data.shape
    new_data = pd.DataFrame(np.zeros((int(n/split) - 1, d)), columns=list(data))
    for i in range(int(n/split) - 1):
        for j in range(int(d/ground_features)):
            # Close
            new_data.iloc[i, j * ground_features] = data.iloc[split * (i + 1), j * ground_features]
            # High
            new_data.iloc[i, j * ground_features + 1] = max([data.iloc[split*i+k, j * ground_features + 1]
                                                             for k in range(split)])
            # Low
            new_data.iloc[i, j * ground_features + 2] = min([data.iloc[split * i + k, j * ground_features + 2]
                                                             for k in range(split)])
            # Open
            new_data.iloc[i, j * ground_features + 3] = data.iloc[split*i, j * ground_features + 3]
            # Volume
            new_data.iloc[i, j * ground_features + 4] = np.sum(data.iloc[i*split:(i+1)*split, j * ground_features + 4])
    return new_data


def preprocess_2_single(data, ticker: str, ground_features: int = 5, new_features: int = 5):
    n, d = data.shape
    new_d = int(d/ground_features)
    new_data = np.zeros((n, new_d * new_features))
    open_prices = np.zeros((n, new_d))
    for i in range(new_d):
        new_data[:, new_features * i] = \
            data.iloc[:, ground_features * i]/data.iloc[:, ground_features * i + 3] - 1  # Returns
        new_data[:, new_features * i + 1] = \
            data.iloc[:, ground_features * i + 1] - data.iloc[:, ground_features * i + 2]  # Spread
        new_data[:, new_features * i + 2] = \
            data.iloc[:, ground_features * i + 4]  # Volume
        new_data[:, new_features * i + 3] = \
            data.iloc[:, ground_features * i + 3]  # Open price
        new_data[:, new_features * i + 4] = \
            np.sin(2 * np.pi * new_data[:, new_features * i + 3]/np.max(new_data[:, new_features * i + 3]))  # Sin

        open_prices[:, i] = data.iloc[:, ground_features * i + 3]
    header_data = []
    header_open = []
    header_data.append(ticker + '_returns')
    header_data.append(ticker + '_spread')
    header_data.append(ticker + '_volume')  # Normalized
    header_data.append(ticker + '_normalized_open')
    header_data.append(ticker + '_sin_returns')
    header_open.append(ticker + '_open')
    return pd.DataFrame(new_data, columns=header_data), pd.DataFrame(open_prices, columns=header_open)


def preprocess_2_multi(data, tickers: list, ground_features: int = 5, new_features: int = 5):
    n, d = data.shape
    new_d = int(d/ground_features)
    new_data = np.zeros((n, new_d * new_features))
    open_prices = np.zeros((n, new_d))
    for i in range(new_d):
        new_data[:, new_features * i] = \
            data.iloc[:, ground_features * i]/data.iloc[:, ground_features * i + 3] - 1  # Returns
        new_data[:, new_features * i + 1] = \
            data.iloc[:, ground_features * i + 1] - data.iloc[:, ground_features * i + 2]  # Spread
        new_data[:, new_features * i + 2] = \
            data.iloc[:, ground_features * i + 4] - np.mean(data.iloc[:, ground_features * i + 4])# Volume
        new_data[:, new_features * i + 3] = \
            data.iloc[:, ground_features * i + 3] - np.mean(data.iloc[:, ground_features * i + 3])  # Open
        new_data[:, new_features * i + 4] = \
            np.sin(2 * np.pi * new_data[:, new_features * i + 3]/np.max(new_data[:, new_features * i + 3]))  # Sin

        open_prices[:, i] = data.iloc[:, ground_features * i + 3]
    header_data = []
    header_open = []
    for ticker in tickers:
        header_data.append(ticker + '_returns')
        header_data.append(ticker + '_spread')
        header_data.append(ticker + '_volume')  # Normalized
        header_data.append(ticker + '_normalized_open')
        header_data.append(ticker + '_sin_returns')
        header_open.append(ticker + '_open')
    return pd.DataFrame(new_data, columns=header_data), pd.DataFrame(open_prices, columns=header_open)


def combine_ts_with_path(base_path, tickers: list):
    stock0 = tickers[0]
    path = base_path+stock0+'.csv'
    data = pd.read_csv(path, index_col="timestamp", parse_dates=True)

    renamer = {'close': stock0+'_close', 'high': stock0+'_high', 'low': stock0+'_low',
               'open': stock0+'_open', 'volume': stock0+'_volume', }
    data = data.rename(columns=renamer)
    tickers.remove(tickers[0])
    for str in tickers:
        path = base_path+str+'.csv'
        new_data = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        renamer = {'close': str+'_close', 'high': str+'_high', 'low': str+'_low',
                   'open': str + '_open', 'volume': str+'_volume', }
        new_data = new_data.rename(columns=renamer)

        data = pd.concat([data, new_data], axis=1, sort=True)
    tickers.insert(0, stock0)
    return data.interpolate()[1:data.shape[0]]
