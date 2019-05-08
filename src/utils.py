import numpy as np


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
        op = 'Open'
        high = 'High'
        low = 'Low'
        close = 'Close'
        vol = 'Volume'
    n, _ = data.shape
    open_price = np.array(data[op])[0:n - 1]
    high_price = np.array(data[high])[0:n - 1]
    low_price = np.array(data[low])[0:n - 1]
    close_price = np.array(data[close])[0:n - 1]
    traded_volume = np.array(data[vol])[0:n - 1]
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
    return new_data[:, 1:]


def lookback_kernel(x, y,
                    period: int = 3):
    n, d = x.shape

