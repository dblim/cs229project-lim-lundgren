import pandas as pd
import numpy as np
from utils import preprocess_arima
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
""" This code is complete shit for now"""


stocks = ['AAP', 'MRK', 'NRG', 'ORLY']
data = pd.read_csv('../data/modified data')
data = data.drop(columns=['Unnamed: 0'])
data, opens = preprocess_arima(data, stocks)
residual_data_train = pd.read_csv('../output/ARIMA_results/VARMAX_train_residuals.csv', parse_dates=True)
residual_data_train = residual_data_train.drop(columns=['Unnamed: 0'])
residual_data_test = pd.read_csv('../output/ARIMA_results/VARMAX_test_residuals.csv', parse_dates=True)
residual_data_test = residual_data_test.drop(columns=['Unnamed: 0'])
residuals = pd.concat((residual_data_train, residual_data_test), axis=0)
n, d = data.shape
n = int(n * 0.85)
train_data = data[0: int(n*0.7)]
test_data = data[int(n*0.7): int(n*0.85)]

# Preprocess
lookback = 24
X = np.zeros((n - lookback, lookback, 2*4))
Y = np.zeros((n - lookback, 4))
for i in range(X.shape[0]):
    X[i, :, 0] = data.iloc[i: (i+lookback), 0]
    X[i, :, 1] = residuals.iloc[i: (i + lookback), 0]
    X[i, :, 2] = data.iloc[i: (i + lookback), 5]
    X[i, :, 3] = residuals.iloc[i: (i + lookback), 1]
    X[i, :, 4] = data.iloc[i: (i + lookback), 10]
    X[i, :, 5] = residuals.iloc[i: (i + lookback), 2]
    X[i, :, 6] = data.iloc[i: (i + lookback), 15]
    X[i, :, 7] = residuals.iloc[i: (i + lookback), 3]
    Y[i, 0] = data.iloc[i + lookback, 0]
    Y[i, 1] = data.iloc[i + lookback, 5]
    Y[i, 2] = data.iloc[i + lookback, 10]
    Y[i, 3] = data.iloc[i + lookback, 15]

X_train = X[0: int(n * 0.7)]
y_train = Y[0: int(n * 0.7)]

X_val = X[int(n*0.7): int(n*0.85)]
y_val = Y[int(n*0.7): int(n*0.85)]

opens_val = opens[int(n*0.7): int(n*0.85)]


# Initialising the RNN
model = Sequential()

# Adding layers. LSTM(25) --> Dropout(0.2) x 4
model.add(LSTM(units=25, return_sequences=True, input_shape=(X_train.shape[1], 8)))
model.add(Dropout(0.2))

model.add(LSTM(units=25, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=25))

# Output layer
model.add(Dense(units=4, activation='linear'))

# Run
model.compile(optimizer='RMSProp', loss='mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs=30, batch_size=96)

predicted_stock_price = model.predict(X_val)

for i, ticker in enumerate(stocks):
    pred = (predicted_stock_price[:, i] + 1) * opens_val.values[:, i]
    real = (y_val[:, i] + 1) * opens_val.values[:, i]
    MSE = sum((pred - real) ** 2) / y_val.shape[0]
    dummy_mse = sum((real[1: real.shape[0]] - real[0: real.shape[0] - 1]) ** 2) / (y_val.shape[0] - 1)
    print('=========', ticker, '=========')
    print('Dummy MSE:', dummy_mse)
    print('MSE:', MSE)
    pred_zero_one = predicted_stock_price[:, i]
    pred_zero_one[pred_zero_one > 0] = 1
    pred_zero_one[pred_zero_one < 0] = 0
    print('Predicted ones:', np.mean(pred_zero_one))
    real_zero_one = y_val[:, i]
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
    print('Dummy guess true rate:', max(np.mean(real_zero_one), 1 - np.mean(real_zero_one)))
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy:', max(accuracy, 1 - accuracy))

    plt.plot(real, color='red', label='Real ' + ticker + ' Stock Price')
    plt.plot(pred, color='blue', label='Predicted ' + ticker + ' Stock Price')
    plt.title(ticker + ' Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(ticker + ' Stock Price')
    plt.legend()
    plt.savefig('../output/R2N2_results/R2N2_first_test_' + ticker + '.png')
    plt.close()
