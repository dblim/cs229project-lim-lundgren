from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import matplotlib.pyplot as plt
ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
data_intraday, _ = ts.get_intraday('GOOGL', interval='1min', outputsize='compact')
data_weekly, _ = ts.get_weekly('GOOGL')

data_intraday['4. close'].plot()
plt.title('Google stock price today')
plt.ylabel('Price')
plt.show()