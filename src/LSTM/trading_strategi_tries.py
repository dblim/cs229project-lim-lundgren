import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tickers = ['ACN', 'AMAT', 'CDNS', 'IBM', 'INTU', 'LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']
ticker = tickers[0]
del tickers[0]

pred_path = '../output/LSTM_results/single_test/' + ticker + '_test_predictions.csv'
real_path = '../output/LSTM_results/single_test/' + ticker + '_test_real.csv'

pred = pd.read_csv(pred_path).values
real = pd.read_csv(real_path).values

treshold = 0

for i, ticker in enumerate(tickers):
    pred_path = '../output/LSTM_results/single_test/' + ticker + '_test_predictions.csv'
    real_path = '../output/LSTM_results/single_test/' + ticker + '_test_real.csv'

    new_pred = pd.read_csv(pred_path).values
    new_real = pd.read_csv(real_path).values

    #pred = np.concatenate((pred, new_pred), axis=1)
    #real = np.concatenate((real, new_real), axis=1)
