from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import keras.backend as K
from keras import optimizers, losses
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
    den = den + 0.01
    return num/den


def lstm_model_mse(stocks: list,
               lookback: int = 24,
               epochs: int = 100,
               batch_size: int,
               learning_rate: float = 0.0002,
               dropout_rate: float = 0.1,
               ground_features: int = 5,
               percentile: int = 10,
               lstm_units : int):
    # Import data
    data = combine_ts(stocks)
    data = minutizer(data, split=5)
    data, _ = preprocess_2_multi(data, stocks)

    # Transform data
    n, d = data.shape
    train_val_test_split = {'train': 0.7, 'val': 0.85, 'test': 1}

    X = np.zeros((n - lookback, lookback, d))
    Y = np.zeros((n - lookback, int(d/ground_features)))
    for i in range(X.shape[0]):
        for j in range(d):
            X[i, :, j] = data.iloc[i:(i+lookback), j]
            if j < int(d/ground_features):
                Y[i, j] = data.iloc[lookback + i, j * ground_features]

    X_train = X[0: int(n * train_val_test_split['train'])]
    y_train = Y[0: int(n * train_val_test_split['train'])]

    X_val = X[int(n*train_val_test_split['train']): int(n*train_val_test_split['val'])]
    y_val = Y[int(n*train_val_test_split['train']): int(n*train_val_test_split['val'])]

    X_test = X[int(n * train_val_test_split['val']): int(n * train_val_test_split['test'])]
    y_test = Y[int(n * train_val_test_split['val']): int(n * train_val_test_split['test'])]

    # Initialising the LSTM
    model = Sequential()

    # Adding layers. LSTM(n) --> Dropout(p)
    model.add(LSTM(units=lstm_units, return_sequences=True, use_bias=True, input_shape=(X_train.shape[1], d)))
    model.add(Dropout(dropout_rate))

    model.add(LSTM(units=int(lstm_units/ground_features), use_bias=False))
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(units=int(lstm_units/ground_features), activation='linear', use_bias=True))

    # Optimizer
    adam_opt = optimizers.adam(lr=learning_rate)

    # Compile
    model.compile(optimizer=adam_opt, loss=customized_loss)

    print(model.summary())

    # Fit
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    # Validate
    predicted_stock_returns = model.predict(X_val)

    # Save
    pd.DataFrame(predicted_stock_returns).to_csv('../output/LSTM_results/valid_results/all_stocks_pred.csv', index=False)
    pd.DataFrame(y_val).to_csv('../output/LSTM_results/valid_results/all_stocks_real.csv', index=False)
    pd.DataFrame(model.predict(X_test)).to_csv('../output/LSTM_results/test_results/all_stocks_pred.csv', index=False)
    pd.DataFrame(y_test).to_csv('../output/LSTM_results/test_results/all_stocks_real.csv', index=False)

    for i, ticker in enumerate(stocks):
        predcted_returns = predicted_stock_returns[:, i].copy()
        actual_returns = y_val[:, i].copy()
        #
        MSE = sum((predcted_returns - actual_returns) ** 2) / y_val.shape[0]
    return MSE

tickers = ['ACN', 'AMAT', 'CDNS', 'IBM', 'INTU', 'LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']

# Search for lstm_units

# 10 numbers sampled  randomly between 100 and 800
lstm_range = [i for i in range(100,800)]
lstm_units = random.sample(lstm_range, 10)

# Search for batch size. Original chose was 96
# 10 numbers sampled  randomly between 50 and 100

batch_size_range = [i for i in range(50,100)]
batch_size = random.sample(batch_size_range, 10)



for units_num in lstm_units:
    for batch_num in batch_size:
        print('MSE:', lstm_model_mse(tickers, num) )
        print('Number of lstm cells:', units_num)
        print('Batch size:', batch_num)
