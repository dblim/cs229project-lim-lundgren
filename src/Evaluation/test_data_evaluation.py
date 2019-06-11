import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import zero_one_loss
import seaborn as sns

tickers = ['ACN', 'AMAT', 'CDNS', 'IBM', 'INTU', 'LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']

LSTM_partial: bool = True
VAR: bool = True
VARMAX: bool = False
R2N2: bool = False

pred_path = '../output/LSTM_results/test_results/partial_all_stocks_pred.csv'
real_path = '../output/LSTM_results/test_results/partial_all_stocks_real.csv'

pred = pd.read_csv(pred_path).values
real = pd.read_csv(real_path).values

if LSTM_partial is True:
    pred_ret = pred[:, 0]
    real_ret = real[:, 0]
    threshold = 0

    strategy_returns = np.zeros(real.shape)
    for i in range(strategy_returns.shape[0]):
        for j in range(strategy_returns.shape[1]):
            if pred[i, j] > threshold:
                strategy_returns[i, j] = real[i, j]
            else:
                strategy_returns[i, j] = -real[i, j]
    strategy_returns = np.mean(strategy_returns, axis=1)
    dummy_returns = np.mean(real, axis=1)
    strat_return_total = 1
    acc_dummy_return = 1
    for i in range(strategy_returns.shape[0]):
        strat_return_total *= (strategy_returns[i] + 1)
        acc_dummy_return *= (dummy_returns[i] + 1)
    for i in range(1, 10):
        pred_ret = np.concatenate((pred_ret, pred[:, i]), axis=0)
        real_ret = np.concatenate((real_ret, real[:, i]), axis=0)
    sns.set(color_codes=True)
    sns.distplot(real_ret, kde=True, hist=True, bins=200, label='Actual returns', hist_kws={'edgecolor':'black'},
                 kde_kws={"lw": 0})
    sns.distplot(pred_ret, kde=True, hist=True, bins=100, label='Predicted returns', hist_kws={'edgecolor':'black'},
                 kde_kws={"lw": 0})
    plt.axis([-0.0125, 0.0125, 0, 275])
    plt.title('LSTM')
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('../output/LSTM_density.png')
    plt.show()
    #plt.close()

    MSE = sum((pred.reshape(int(pred.shape[0]*pred.shape[1]), 1) -
               real.reshape(int(real.shape[0]*real.shape[1]), 1))**2)/int(pred.shape[0]*pred.shape[1])

    real_zero_one = real.reshape(real.shape[0] * real.shape[1], 1)
    pred_zero_one = pred.reshape(real.shape[0] * real.shape[1], 1)
    real_zero_one[real_zero_one > 0] = 1
    real_zero_one[real_zero_one < 0] = 0

    pred_zero_one[pred_zero_one > 0] = 1
    pred_zero_one[pred_zero_one < 0] = 0

    print('Dummy MSE:', sum(sum(real**2))/(real.shape[0]*real.shape[1]))
    print('Dummy Accuracy:', np.mean(real_zero_one))
    print('Dummy return:', (acc_dummy_return - 1) * 100)
    print('Dummy SR:', np.mean(dummy_returns) / np.std(dummy_returns))
    print('=====')

    print('LSTM MSE:', MSE)
    print('LSTM Accuracy:', 1 - zero_one_loss(real_zero_one, pred_zero_one))
    print('LSTM return:', (strat_return_total - 1) * 100)
    print('LSTM SR:', np.mean(strategy_returns)/np.std(strategy_returns))

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
    VARMAX_threshold = 0
    val_path1 = '../output/VARMAX_results/test_files/'
    ## Name below is REAL becuase I accidentaly named the predictions real and the real ones predictions in the ARIMA file
    val_path_pred = '_test_real.csv' # real = pred... mistake in ARIMA file
    pred_returns = np.zeros(real.shape)
    for i, ticker in enumerate(tickers):
        path = val_path1 + ticker + val_path_pred
        preds = pd.read_csv(path).values
        # Remove the first lookback points in order to give a fair comparison (i.e. same size of arrays)
        preds = preds[preds.shape[0]-pred.shape[0]: preds.shape[0]]
        pred_returns[:, i] = preds.reshape(pred.shape[0], )

    sns.set(color_codes=True)
    sns.distplot(real.reshape(int(real.shape[0]*real.shape[1]), 1), kde=True, hist=True, bins=200,
                 label='Actual returns', hist_kws={'edgecolor': 'black'}, kde_kws={"lw": 0})
    sns.distplot(pred_returns.reshape(int(real.shape[0]*real.shape[1]), 1), kde=True, hist=True, bins=100,
                 label='Predicted returns', hist_kws={'edgecolor': 'black'}, kde_kws={"lw": 0})
    plt.axis([-0.0125, 0.0125, 0, 500])
    plt.title('VARMAX')
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('../output/VARMAX_density.png')
    plt.show()

    # Metrics
    # MSE
    MSE = sum(sum((pred_returns - real)**2))/(real.shape[0] * real[1])
    # accuracy
    VARMAX_zero_one = pred_returns.reshape(int(real.shape[0]*real.shape[1]), 1)
    VARMAX_zero_one[VARMAX_zero_one > 0] = 1
    VARMAX_zero_one[VARMAX_zero_one < 0] = 0
    # returns
    VARMAX_strategy_returns = np.zeros(real.shape)
    for i in range(real.shape[0]):
        for j in range(real.shape[1]):
            if pred_returns[i, j] > VARMAX_threshold:
                VARMAX_strategy_returns[i, j] = real[i, j]
            else:
                VARMAX_strategy_returns[i, j] = -real[i, j]
    VARMAX_strategy_returns = np.mean(VARMAX_strategy_returns, axis=1)
    VARMAX_strategy_ret_total = 1
    for i in range(VARMAX_strategy_returns.shape[0]):
        VARMAX_strategy_ret_total *= (1 + VARMAX_strategy_returns[i])

    print('=====')
    print('VARMAX MSE:', MSE)
    print('VARMAX Accuracy:', 1 - zero_one_loss(real_zero_one, VARMAX_zero_one))
    print('VARMAX return:', (VARMAX_strategy_ret_total - 1) * 100)
    print('VARMAX SR:', np.mean(VARMAX_strategy_returns)/np.std(VARMAX_strategy_returns))

if VAR is True:
    var_test_path = '../output/VAR_results/VAR_test_predictions.csv'
    var_data = pd.read_csv(var_test_path)
    var_data = var_data.drop(columns=['Unnamed: 0']).values

    threshold = 0

    strategy_returns = np.zeros(real.shape)
    print(strategy_returns.shape)

    for i in range(strategy_returns.shape[0] - 1):
        for j in range(strategy_returns.shape[1]):
            if var_data[i + 1, j] > threshold:
                strategy_returns[i, j] = real[i, j]
            else:
                strategy_returns[i, j] = -real[i, j]
    strategy_returns = np.mean(strategy_returns, axis=1)

    strat_return_total = 1
    for i in range(strategy_returns.shape[0]):
        strat_return_total *= (strategy_returns[i] + 1)

    sns.set(color_codes=True)
    sns.distplot(real.reshape(real.shape[0] * real.shape[1], 1), kde=True, hist=True, bins=200, label='Actual returns',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={"lw": 0})
    sns.distplot(var_data.flatten(), kde=True, hist=True, bins=100,
                 label='Predicted returns', hist_kws={'edgecolor': 'black'},
                 kde_kws={"lw": 0})
    plt.axis([-0.0125, 0.0125, 0, 2500])
    plt.title('VAR')
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('../output/VAR_density.png')
    plt.show()
    # plt.close()

    MSE = sum(sum((var_data.reshape(var_data.shape[0]*var_data.shape[1], 1) -
                   real.reshape(real.shape[0]*real.shape[1], 1))**2))/ int(pred.shape[0] * pred.shape[1])

    pred_zero_one = var_data.flatten()
    pred_zero_one[pred_zero_one > 0] = 1
    pred_zero_one[pred_zero_one < 0] = 0

    print('=====')
    print('VAR MSE:', MSE)
    print('VAR Accuracy:', 1 - zero_one_loss(real_zero_one, pred_zero_one))
    print('VAR return:', (strat_return_total - 1) * 100)
    print('VAR SR:', np.mean(strategy_returns) / np.std(strategy_returns))


if R2N2 is True:
    threshold = 0
    R2N2_path = '../output/R2N2_results/test_all_stocks_pred.csv'
    R2N2_data = pd.read_csv(R2N2_path).values

    strategy_returns = np.zeros(real.shape)

    for i in range(strategy_returns.shape[0]):
        for j in range(strategy_returns.shape[1]):
            if R2N2_data[i, j] > threshold:
                strategy_returns[i, j] = real[i, j]
            else:
                strategy_returns[i, j] = -real[i, j]
    strategy_returns = np.mean(strategy_returns, axis=1)

    strat_return_total = 1
    for i in range(strategy_returns.shape[0]):
        strat_return_total *= (strategy_returns[i] + 1)



    sns.set(color_codes=True)
    sns.distplot(real.reshape(real.shape[0]*real.shape[1], 1), kde=True, hist=True, bins=200, label='Actual returns', hist_kws={'edgecolor': 'black'},
                 kde_kws={"lw": 0})
    sns.distplot(R2N2_data.reshape(real.shape[0]*real.shape[1], 1), kde=True, hist=True, bins=100, label='Predicted returns', hist_kws={'edgecolor': 'black'},
                 kde_kws={"lw": 0})
    plt.axis([-0.0125, 0.0125, 0, 5000])
    plt.title('R2N2')
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('../output/R2N2_density.png')
    plt.show()
    # plt.close()

    MSE = sum((R2N2_data.reshape(int(pred.shape[0] * pred.shape[1]), 1) -
               real.reshape(int(real.shape[0] * real.shape[1]), 1)) ** 2) / int(pred.shape[0] * pred.shape[1])

    #real_zero_one = real.reshape(real.shape[0] * real.shape[1], 1)
    pred_zero_one = R2N2_data.reshape(real.shape[0] * real.shape[1], 1)
    pred_zero_one[pred_zero_one > 0] = 1
    pred_zero_one[pred_zero_one < 0] = 0

    print('=====')
    print('R2N2 MSE:', MSE)
    print('R2N2 Accuracy:', 1 - zero_one_loss(real_zero_one, pred_zero_one))
    print('R2N2 return:', (strat_return_total - 1) * 100)
    print('R2N3 SR:', np.mean(strategy_returns) / np.std(strategy_returns))
