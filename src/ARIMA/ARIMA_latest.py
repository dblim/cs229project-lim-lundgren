from statsmodels.tsa.statespace.varmax import VARMAX
#from utils import combine_ts, minutizer, preprocess_2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def combine_ts(tickers: list):
    stock0 = tickers[0]
    path = '../data/sectors/Information Technology/'+stock0+'.csv'
    data = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    renamer = {'close': stock0+'_close', 'high': stock0+'_high', 'low': stock0+'_low',
               'open': stock0+'_open', 'volume': stock0+'_volume', }
    data = data.rename(columns=renamer)
    tickers.remove(tickers[0])
    for str in tickers:
        path = '../data/sectors/Information Technology/'+str+'.csv'
        new_data = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        renamer = {'close': str+'_close', 'high': str+'_high', 'low': str+'_low',
                   'open': str + '_open', 'volume': str+'_volume', }
        new_data = new_data.rename(columns=renamer)

        data = pd.concat([data, new_data], axis=1, sort=True)
    tickers.insert(0, stock0)
    return data.interpolate()[1:data.shape[0]]


def minutizer(data, split: int = 5, ground_features: int = 5):
    n, d = data.shape
    new_data = pd.DataFrame(np.zeros((int(n/split) - 1, d)), columns=list(data))
    for i in range(int(n/split) - 1):
        for j in range(int(d/ground_features)):
            # Close
            new_data.iloc[i, j * ground_features] = data.iloc[split * (i + 1), j * ground_features]
            # High
            new_data.iloc[i, j * ground_features + 1] = max([data.iloc[split*i+k, j * ground_features + 1]
                                                             for k in range(split)])
            # Low
            new_data.iloc[i, j * ground_features + 2] = min([data.iloc[split * i + k, j * ground_features + 2]
                                                             for k in range(split)])
            # Open
            new_data.iloc[i, j * ground_features + 3] = data.iloc[split*i, j * ground_features + 3]
            # Volume
            new_data.iloc[i, j * ground_features + 4] = np.sum(data.iloc[i*split:(i+1)*split, j * ground_features + 4])
    return new_data


def preprocess_2(data, tickers, ground_features: int = 5, new_features: int = 5):
    n, d = data.shape
    new_d = int(d/ground_features)
    new_data = np.zeros((n, new_d * new_features))
    open_prices = np.zeros((n, new_d))
    for i in range(new_d):
        new_data[:, new_features * i] = \
            data.iloc[:, ground_features * i]/data.iloc[:, ground_features * i + 3] - 1  # Returns
        new_data[:, new_features * i + 1] = \
            data.iloc[:, ground_features * i + 1] - data.iloc[:, ground_features * i + 2]  # Spread
        new_data[:, new_features * i + 2] = \
            data.iloc[:, ground_features * i + 4] - np.mean(data.iloc[:, ground_features * i + 4])# Volume
        new_data[:, new_features * i + 3] = \
            data.iloc[:, ground_features * i + 3] - np.mean(data.iloc[:, ground_features * i + 3])  # Open
        # Laguerre polynomials
        """
        new_data[:, new_features * i + 3] = np.exp(- 0.5 * new_data[:, new_features * i + 3])
        new_data[:, new_features * i + 4] = new_data[:, new_features * i + 4] * (1 - new_data[:, new_features * i + 3])
        new_data[:, new_features * i + 5] = new_data[:, new_features * i + 4] * \
            (1 - 2 * new_data[:, new_features * i + 3] + 0.5 * np.square(new_data[:, new_features * i + 3]))
        """
        new_data[:, new_features * i + 4] = \
            np.sin(2 * np.pi * new_data[:, new_features * i + 3]/np.max(new_data[:, new_features * i + 3]))  # Sin

        open_prices[:, i] = data.iloc[:, ground_features * i + 3]
    header_data = []
    for ticker in tickers:
        header_data.append(ticker + '_returns')  # 0
        header_data.append(ticker + '_spread')  # 1
        header_data.append(ticker + '_volume')  # 2
        header_data.append(ticker + '_open')  # 3
        header_data.append(ticker + '_sin_open')  # 8
    return pd.DataFrame(new_data, columns=header_data)

tickers = ['ACN', 'AMAT', 'CDNS', 'IBM', 'INTU', 'LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']

data = preprocess_2(minutizer(combine_ts(tickers), split=5), tickers)
n, _ = data.shape

def varmax(tickers,
           p: int = 2,
           q: int = 0,):
    #data = preprocess_2(minutizer(combine_ts(tickers), split=5), tickers)
    #n, _ = data.shape

    # Split data
    train_val_test_split = {'train': 0.7, 'val': 0.85, 'test': 1}
    train_data = data[0: int(n*train_val_test_split['train'])]
    val_data = data[int(n*train_val_test_split['train']): int(n*train_val_test_split['val'])]
    test_data = data[int(n * train_val_test_split['val']): int(n * train_val_test_split['test'])]

    # split data in X and Y
    y_list = [ticker+'_returns' for ticker in tickers]

    # Train
    endog_y = train_data[y_list]
    exog_x = train_data.drop(columns=y_list)

    # Validate
    endog_y_val = val_data[y_list]
    exog_x_val = val_data.drop(columns=y_list)

    # Test
    endog_y_test = test_data[y_list]
    exog_x_test = test_data.drop(columns=y_list)

    # Fit model
    model = VARMAX(endog=endog_y.values, exog=exog_x.values, order=(p, q))
    model_fit = model.fit(disp=False, order=(p, q), maxiter=200, method='nm')
    # Validate
    predictions_val = model_fit.forecast(steps=exog_x_val.shape[0], exog=exog_x_val.values)
    MSE = 0
    for i in range(endog_y_val.shape[0]):
        for j in range(endog_y_val.shape[1]):
            MSE += (endog_y_val.values[i, j] - float(predictions_val[i][j]))**2
    print('p:', p, ' MSE:', MSE)
    '''
    # Test -- this is just here for simplcity!!
    predictions_test = model_fit.forecast(steps=exog_x_test.shape[0], exog=exog_x_test.values)

    for i, ticker in enumerate(tickers):
        real_val = endog_y_val.values[:, i]
        pred_val = predictions_val[:, i]
        pd.DataFrame(real_val).to_csv('../output/ARIMA_results/predictions/val_files/'+ticker+'_val_predictions.csv',
                                      index=False)
        pd.DataFrame(pred_val).to_csv('../output/ARIMA_results/predictions/val_files/' + ticker + '_val_real.csv',
                                      index=False)
        real_test = endog_y_test.values[:, i]
        pred_test = predictions_test[:, i]
        pd.DataFrame(real_test).to_csv('../output/ARIMA_results/predictions/test_files/' + ticker + '_test_predictions.csv',
                                       index=False)
        pd.DataFrame(pred_test).to_csv('../output/ARIMA_results/predictions/test_files/' + ticker + '_test_real.csv',
                                       index=False)
    '''
    # Evaluate
    pic: bool = False
    if pic is True:
        for i, ticker in enumerate(tickers):
            pred = (predictions[:, i] + 1) * opens_val.values[:, i]
            real = (endog_y_val.values[:, i] + 1) * opens_val.values[:, i]
            MSE = sum((pred - real)**2)/endog_y_val.shape[0]
            dummy_mse = sum((real[1: real.shape[0]] - real[0: real.shape[0] - 1]) ** 2) / (endog_y_val.shape[0] - 1)
            print('=========', ticker, '=========')
            print('Dummy MSE:', dummy_mse)
            print('MSE:', MSE)
            pred_zero_one = predictions[:, i]
            pred_zero_one[pred_zero_one > 0] = 1
            pred_zero_one[pred_zero_one < 0] = 0
            print('Predicted ones:', np.mean(pred_zero_one))
            real_zero_one = endog_y_val.values[:, i]
            real_zero_one[real_zero_one > 0] = 1
            real_zero_one[real_zero_one < 0] = 0
            print('Real ones:', np.mean(real_zero_one))
            TP = np.sum(np.logical_and(pred_zero_one == 1, real_zero_one == 1))
            TN = np.sum(np.logical_and(pred_zero_one == 0, real_zero_one == 0))
            FP = np.sum(np.logical_and(pred_zero_one == 1, real_zero_one == 0))
            FN = np.sum(np.logical_and(pred_zero_one == 0, real_zero_one == 1))
            print('True positive:', TP)
            print('True Negative:', TN)
            print('False positive:', FP)
            print('False Negative:', FN)
            accuracy = (TP + TN)/(TP + TN + FP + FN)
            print('Dummy guess:', max(np.mean(real_zero_one), 1 - np.mean(real_zero_one)))
            print('Accuracy:', max(accuracy, 1 - accuracy))

            plt.plot(real, color='red', label='Real ' + ticker + ' Stock Price')
            plt.plot(pred, color='blue', label='Predicted ' + ticker + ' Stock Price')
            plt.title(ticker + ' Stock Price Prediction')
            plt.xlabel('Time')
            plt.ylabel(ticker + ' Stock Price')
            plt.legend()
            plt.savefig('../output/ARIMA_results/VARMAX_test_' + ticker + '.png')
            plt.close()


for i in range(50):
    varmax(tickers, p=1, q=i)