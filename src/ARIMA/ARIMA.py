from statsmodels.tsa.statespace.varmax import VARMAX
from utils import combine_ts, minutizer, preprocess_arima
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get data
tickers = ['AAP', 'MRK', 'NRG', 'ORLY']
data, opens = preprocess_arima(minutizer(combine_ts(tickers)), tickers)
n, _ = data.shape

# Split data
train_val_test_split = {'train': 0.7, 'val': 0.85, 'test': 1}
train_data = data[0: int(n*train_val_test_split['train'])]
val_data = data[int(n*train_val_test_split['train']): int(n*train_val_test_split['val'])]

opens_val = opens[int(n*train_val_test_split['train']): int(n*train_val_test_split['val'])]

# split data in X and Y
y_list = [ticker+'_returns' for ticker in tickers]
# Train
endog_y = train_data[y_list]
exog_x = train_data.drop(columns=y_list)
# Validate
endog_y_val = val_data[y_list]
exog_x_val = val_data.drop(columns=y_list)

# Fit model
model = VARMAX(endog=endog_y.values, exog=exog_x.values, order=(2, 2))
model_fit = model.fit(disp=False)

# make prediction
predictions = model_fit.forecast(steps=exog_x_val.shape[0], exog=exog_x_val.values)

residuals = np.subtract(np.array(predictions).reshape(endog_y_val.shape[0], len(tickers)), endog_y_val.values)
residuals = pd.DataFrame(residuals, columns=tickers)
residuals.to_csv('../output/ARIMA_results/VARMAX_test_residuals.csv')

# Evaluate
for i, ticker in enumerate(tickers):
    pred = (predictions[:, i] + 1) * opens_val.values[:, i]
    real = (endog_y_val.values[:, i] + 1) * opens_val.values[:, i]
    MSE = sum((pred - real)**2)/endog_y_val.shape[0]
    print('=========', ticker, '=========')
    print(' MSE:', MSE)
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
