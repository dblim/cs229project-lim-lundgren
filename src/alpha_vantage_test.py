from alpha_vantage.timeseries import TimeSeries
from utils import preprocess, lookback_kernel
ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
data, _ = ts.get_intraday('^GSPC', outputsize='full')

print(data.shape)