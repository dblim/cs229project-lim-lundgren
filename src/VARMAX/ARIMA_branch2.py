from statsmodels.tsa.statespace.varmax import VARMAX
from utils import combine_ts, minutizer, preprocess_2_multi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tickers = ['ACN', 'AMAT', 'CDNS', 'IBM', 'INTU', 'LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']

#data, _ = preprocess_2_multi(minutizer(combine_ts(tickers), split=5), tickers)

data = pd.read_csv('../data/preprocessed_time_series_data.csv')
data = data.drop(columns=['Unnamed: 0'])
new_data = np.zeros((data.shape[0], 10))
n, _ = data.shape


def varmax(tickers,
           p: int = 2,
           q: int = 0,):

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
    #for i in range(endog_y_val.shape[0]):
    #    for j in range(endog_y_val.shape[1]):
    #        MSE += (endog_y_val.values[i, j] - float(predictions_val[i][j]))**2
    print('p:', p, ' MSE:', MSE)

    # Test -- this is just here for simplcity!!
    predictions_test = model_fit.forecast(steps=exog_x_test.shape[0], exog=exog_x_test.values)

    train_residuals = model_fit.resid
    pd.DataFrame(train_residuals).to_csv('../output/VARMAX_results/residual_data_train.csv')
    val_residual = endog_y_val.values - predictions_val
    pd.DataFrame(val_residual).to_csv('../output/VARMAX_results/residual_data_val.csv')
    test_residual = endog_y_test.values - predictions_test
    pd.DataFrame(test_residual).to_csv('../output/VARMAX_results/residual_data_test.csv')

    q: bool = False
    if q is True:
        for i, ticker in enumerate(tickers):
            real_val = endog_y_val.values[:, i]
            pred_val = predictions_val[:, i]
            pd.DataFrame(real_val).to_csv('../output/VARMAX_results/val_files/'+ticker+'_val_predictions.csv',
                                          index=False)
            pd.DataFrame(pred_val).to_csv('../output/VARMAX_results/val_files/' + ticker + '_val_real.csv',
                                          index=False)
            real_test = endog_y_test.values[:, i]
            pred_test = predictions_test[:, i]
            pd.DataFrame(real_test).to_csv('../output/VARMAX_results/test_files/' + ticker + '_test_predictions.csv',
                                           index=False)
            pd.DataFrame(pred_test).to_csv('../output/VARMAX_results/test_files/' + ticker + '_test_real.csv',
                                           index=False)

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
            plt.savefig('../output/VARMAX_results/VARMAX_test_' + ticker + '.png')
            plt.close()

varmax(tickers, p=1, q=2)
