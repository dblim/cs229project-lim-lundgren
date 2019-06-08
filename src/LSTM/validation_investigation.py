import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tickers = ['ACN', 'AMAT', 'CDNS', 'IBM', 'INTU', 'LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']

pred_path = '../output/LSTM_results/valid_results/all_stocks_pred.csv'
real_path = '../output/LSTM_results/valid_results/all_stocks_real.csv'

pred = pd.read_csv(pred_path).values
real = pd.read_csv(real_path).values

pred_ret = np.zeros((int(pred.shape[0]/10), 10))
real_ret = np.zeros((int(pred.shape[0]/10), 10))

for i in range(int(pred.shape[0]/10)):
    for j in range(10):
        idx = 10 * i + j
        pred_ret[i, j] = pred[idx]
        real_ret[i, j] = real[idx]

plt.hist(pred, bins=50, density=False, label='real')
plt.hist(real, bins=50, density=False, label='pred', alpha=0.75)
plt.legend()
plt.show()

kuk: bool = False
if kuk is True:
    for i in range(10):
        pred_i = pred_ret[:, i]*10
        real_i = real_ret[:, i]

        plt.hist(real_i, bins=50, density=True, label='real')
        plt.hist(pred_i, bins=50, density=True, label='pred', alpha=0.75)
        plt.legend()
        plt.show()