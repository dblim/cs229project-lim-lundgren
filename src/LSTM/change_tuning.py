from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import keras.backend as K
from keras import optimizers
import numpy as np
import pandas as pd
import random
from lstm_utils import minutizer, combine_ts, preprocess_2_multi, customized_loss

def lstm_model_mse(lstm_units :int, batch_size : int, stocks: list,
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


    # Initialising the LSTM
    model = Sequential()

    # Adding layers. LSTM(n) --> Dropout(p)
    model.add(LSTM(units=lstm_units, return_sequences=True, use_bias=True, input_shape=(lookback, ground_features)))
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
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose =0)

    # Validate
    predicted_stock_returns = model.predict(X_val)

    all_mse = []
    for i, ticker in enumerate(stocks):
        predcted_returns = np.zeros((int(y_val.shape[0] / amount_of_stocks), 1))
        actual_returns = np.zeros((int(y_val.shape[0] / amount_of_stocks), 1))
        for j in range(int(y_val.shape[0] / amount_of_stocks)):
            predcted_returns[j] = predicted_stock_returns[amount_of_stocks * j + i]
            actual_returns[j] = y_val[amount_of_stocks * j + i]
        #
        MSE = sum((predcted_returns - actual_returns) ** 2) / y_val.shape[0]
        all_mse.append(MSE)

    avg_mse = np.array(all_mse)
    print('Average MSE:', np.mean(avg_mse))

tickers = ['ACN', 'AMAT' ,  'CDNS', 'IBM', 'INTU', 'LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']

# Search lstm size between 10 and 50
# Search batch size between 50 and 100.
random.seed()

#Choose 10 random random numbers
for k in range(10):
    lstm_units, batch_size = random.randint(10, 50), random.randint(80, 120)
    print('Number of LSTM cells:', lstm_units)
    print('Batch size:', batch_size)
    lstm_model_mse(lstm_units, batch_size, tickers)