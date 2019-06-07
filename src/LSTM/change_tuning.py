from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import keras.backend as K
from keras import optimizers
import numpy as np
import pandas as pd
import random
from lstm_utils import minutizer, combine_ts, preprocess_2_multi


def customized_loss(y_pred, y_true):
    num = K.sum(K.square(y_pred - y_true), axis=-1)
    y_true_sign = y_true > 0
    y_pred_sign = y_pred > 0
    logicals = K.equal(y_true_sign, y_pred_sign)
    logicals_0_1 = K.cast(logicals, 'float32')
    den = K.sum(logicals_0_1, axis=-1)
    return num/(1 + den)


def lstm_model_mse(lstm_units :list, batch_size : list, stocks: list,
               lookback: int = 24,
               epochs: int = 2,
               learning_rate: float = 0.0001,
               dropout_rate: float = 0.1,
               ground_features: int = 4,
               percentile: int = 10):
    # Import data
    data = combine_ts(stocks)
    data = minutizer(data, split=5)
    data, _ = preprocess_2_multi(data, stocks)

    # Transform data
    n, d = data.shape
    amount_of_stocks = int(d/ground_features)
    train_val_test_split = {'train': 0.7, 'val': 0.85, 'test': 1}

    new_n = (n - lookback) * amount_of_stocks

    X = np.zeros((new_n, lookback, ground_features))
    Y = np.zeros((new_n, 1))

    for i in range(n - lookback):
        for j in range(amount_of_stocks):
            idx = i * amount_of_stocks + j
            for k in range(ground_features):
                col = j * ground_features + k
                X[idx, :, k] = data.iloc[i: (i + lookback), col]
            Y[idx] = data.iloc[i + lookback, ground_features * j]

    X_train = X[0: int(new_n * train_val_test_split['train'])]
    y_train = Y[0: int(new_n * train_val_test_split['train'])]

    X_val = X[int(new_n * train_val_test_split['train']): int(new_n * train_val_test_split['val'])]
    y_val = Y[int(new_n * train_val_test_split['train']): int(new_n * train_val_test_split['val'])]

    X_test = X[int(new_n * train_val_test_split['val']): int(new_n * train_val_test_split['test'])]
    y_test = Y[int(new_n * train_val_test_split['val']): int(new_n * train_val_test_split['test'])]

    # Hyperparameter printing
    for units_num in lstm_units:
        for batch_num in batch_size:

            # Initialising the LSTM
            model = Sequential()

            # Adding layers. LSTM(n) --> Dropout(p)
            model.add(LSTM(units=units_num, return_sequences=True, use_bias=True, input_shape=(lookback, ground_features)))
            model.add(Dropout(dropout_rate))

            model.add(LSTM(units=10, use_bias=False))
            model.add(Dropout(dropout_rate))

            # Output layer
            model.add(Dense(units=1, activation='linear', use_bias=True))

            # Optimizer
            adam_opt = optimizers.adam(lr=learning_rate)

            # Compile
            model.compile(optimizer=adam_opt, loss=customized_loss)

            # Fit
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_num, validation_data=(X_val, y_val))

            # Validate
            predicted_stock_returns = model.predict(X_val)

            for i, ticker in enumerate(stocks):
                predcted_returns = np.zeros((int(y_val.shape[0] / amount_of_stocks), 1))
                actual_returns = np.zeros((int(y_val.shape[0] / amount_of_stocks), 1))
                for j in range(int(y_val.shape[0] / amount_of_stocks)):
                    predcted_returns[j] = predicted_stock_returns[amount_of_stocks * j + i]
                    actual_returns[j] = y_val[amount_of_stocks * j + i]
                #
                MSE = sum((predcted_returns - actual_returns) ** 2) / y_val.shape[0]

            print('MSE:', MSE)
            print('Number of LSTM cells:', units_num)
            print('Batch size:', batch_num)


# Search for lstm_units

# 10 numbers sampled  randomly between 100 and 800
lstm_range = [i for i in range(10,50)]
lstm_units = random.sample(lstm_range, 5)

# Search for batch size. Original chose was 96
# 10 numbers sampled  randomly between 50 and 100

batch_size_range = [i for i in range(90,100)]
batch_size = random.sample(batch_size_range, 5)

tickers = ['ACN', 'AMAT'] # 'CDNS', 'IBM', 'INTU', 'LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']

lstm_model_mse(lstm_units, batch_size,tickers)