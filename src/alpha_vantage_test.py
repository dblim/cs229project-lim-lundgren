from alpha_vantage.timeseries import TimeSeries
ts = TimeSeries(key='YOUR_API_KEY')
data, _ = ts.get_intraday('GOOGL')

print(data)