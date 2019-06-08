import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tickers = ['ACN', 'AMAT', 'CDNS', 'IBM', 'INTU', 'LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']

LSTM: bool = True
VARMAX: bool = True

pred_path = '../output/LSTM_results/test_results/partial_all_stocks_pred.csv'
real_path = '../output/LSTM_results/test_results/partial_all_stocks_real.csv'

pred = pd.read_csv(pred_path).values
real = pd.read_csv(real_path).values
if LSTM is True:
    pred_ret = pred[:, 0]
    real_ret = real[:, 0]
    corr = np.corrcoef(real_ret, pred_ret)[0][1]
    for i in range(1, 10):
        #print(abs(np.corrcoef(pred[:, i], real[:, i])[0][1]))
        pred_ret = np.concatenate((pred_ret, pred[:, i]), axis=0)
        real_ret = np.concatenate((real_ret, real[:, i]), axis=0)
        corr += abs(np.corrcoef(real_ret, pred_ret)[0][1])

    plt.hist(real_ret, bins=200, density=True, label='real')
    plt.hist(pred_ret, bins=200, density=True, label='pred', alpha=0.75)
    plt.axis([-0.01, 0.01, 0, 700])
    plt.title('LSTM')
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    MSE = sum((pred.reshape(int(pred.shape[0]*pred.shape[1]), 1) -
               real.reshape(int(real.shape[0]*real.shape[1]), 1))**2)/int(pred.shape[0]*pred.shape[1])

    print('LSTM MSE:', MSE)
    print('LSTM correlation:', corr/10)

    inc: bool = False
    if inc is True:
        for i in range(10):
            pred_i = pred[:, i]
            real_i = real[:, i]

            plt.hist(real_i, bins=50, density=True, label='real')
            plt.hist(pred_i, bins=50, density=True, label='pred', alpha=0.75)
            plt.legend()
            plt.show()


if VARMAX is True:
    val_path1 = '../output/VARMAX_results/val_files/'
    val_path_pred = '_val_predictions.csv'
    pred_returns = np.zeros((0, 1))
    corr = 0
    for i, ticker in enumerate(tickers):
        path = val_path1 + ticker + val_path_pred
        preds = pd.read_csv(path).values
        preds = preds[preds.shape[0]-pred.shape[0]: preds.shape[0]]
        pred_returns = np.concatenate((pred_returns, preds))
        corr += abs(np.corrcoef(preds.T, real[:, i].reshape(1265, 1).T)[0][1])
    plt.hist(real.reshape(int(real.shape[0]*real.shape[1]), 1), bins=200, density=True, label='real')
    plt.hist(pred_returns, bins=200, density=True, label='pred', alpha=0.75)
    plt.title('VARMAX')
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.axis([-0.01, 0.01, 0, 700])
    plt.legend()
    plt.show()

    # Metrics

    MSE = sum((real.reshape(int(real.shape[0]*real.shape[1]), 1) - pred_returns)**2)/pred_returns.shape[0]
    print('VARMAX MSE:', MSE)
    print(corr/10)


