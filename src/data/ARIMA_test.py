import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot as plt

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


ticker = 'NVDA'
stock_df = read_stock(ticker)
stock_df = append_returns(stock_df)

# this value picked based on auto_correlation plot
start_time = '2017-09-15 14:05:00'

# size of test and train data
train = 40
test = 40

ts = stock_df[['returns']]
start_index = ts.index.get_loc(start_time)

predictions = []
for i in range(test):
    # choose model and fit
    model = ARIMA(ts.iloc[start_index+i:start_index+i+train],order=(2,0,0))
    model_fit = model.fit(disp=0)
    yhat = model_fit.forecast()[0]
    predictions.append(yhat[0])

# plot the predictions
# create new data frame with predictions
# to be cleaned up later
top_df = pd.DataFrame(ts.index[start_index+train-10:start_index+train])
top_df['returns'] =  list(ts['returns'][start_index+train-10:start_index+train])
top_df['predictions'] = list(ts['returns'][start_index+train-10:start_index+train])
top_df = top_df.set_index('timestamp')

bot_df = pd.DataFrame(ts.index[start_index+train:start_index+train+20])
bot_df['returns'] = list(ts['returns'][start_index+train:start_index+train+20])
bot_df['predictions'] = [x*3 for x in predictions[:20]]
bot_df = bot_df.set_index('timestamp')


tot_df = pd.concat([top_df, bot_df])
m = len(tot_df)
tot_df['zero'] = [0 for i in range(m)]
tot_df.plot()
plt.title('ARIMA model for '+ticker)
plt.savefig('ARIMA_test_'+ticker)
