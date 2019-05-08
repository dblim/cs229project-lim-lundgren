from alpha_vantage.timeseries import TimeSeries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
data, _ = ts.get_intraday('GOOGL', outputsize='full')

#data['4. close'].plot()
#plt.show()
print(data)
print(np.array(data['1. open']))