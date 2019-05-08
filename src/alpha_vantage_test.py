from alpha_vantage.timeseries import TimeSeries
from utils import preprocess, lookback_kernel
ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
data, _ = ts.get_intraday('^GSPC', outputsize='full')

x, y = preprocess(data, incremental_data=True)

print(x)
print(x.shape)

x, y = lookback_kernel(x, y)

print(x)
print(x.shape)