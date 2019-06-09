import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import zero_one_loss
import seaborn as sns

tickers = ['ACN', 'AMAT', 'CDNS', 'IBM', 'INTU', 'LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']

pred_path_LSTM = '../output/LSTM_results/test_results/partial_all_stocks_pred.csv'
pred_path_varmax = '../output/VARMAX_results/test_files/'
real_path = '../output/LSTM_results/test_results/partial_all_stocks_real.csv'

pred_LSTM = pd.read_csv(pred_path_LSTM).values
real = pd.read_csv(real_path).values
pred_VARMAX = np.zeros(pred_LSTM.shape)
for i, ticker in enumerate(tickers):
    path = pred_path_varmax+ticker+'_test_real.csv'  # THis is cuz i failed to name the VARMAX correctly
    partial_varmax_data = pd.read_csv(path).values
    partial_varmax_data = partial_varmax_data[24: partial_varmax_data.shape[0]]
    pred_VARMAX[:, i] = partial_varmax_data.reshape(1265, )


real_path = [1]
pred_path_LSTM = [1]
pred_path_VARMAX = [1]
dummy = []

pred_timead_path_LSTM = [1]
pred_timead_path_varmax = [1]
real_timead_path = [1]
dummy_ta = [1]

time_scalar = 12
for i in range(pred_LSTM.shape[0]):
    real_return = np.mean(real[i, :])
    pred_return_LSTM = np.mean(pred_LSTM[i, :])
    pred_return_VARMAX = np.mean(pred_VARMAX[i, :])

    real_path.append((real_return + 1) * real_path[i - 1])
    pred_path_LSTM.append((pred_return_LSTM + 1) * real_path[i - 1])
    pred_path_VARMAX.append((pred_return_VARMAX + 1) * real_path[i - 1])
    dummy.append(real_path[i-1])
    if i % time_scalar == 0:
        real_timead_path.append(real_path[i])
        pred_timead_path_LSTM.append(pred_path_LSTM[i])
        pred_timead_path_varmax.append(pred_path_VARMAX[i])
        dummy_ta.append(dummy[i])

plt.plot(dummy_ta, label='dummy')
plt.plot(pred_timead_path_LSTM, label='LSTM pred')
plt.plot(pred_timead_path_varmax, label='VARMAX pred')
plt.plot(real_timead_path, label='real')

plt.legend()
plt.show()