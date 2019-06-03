from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import optimizers
import numpy as np
import pandas as pd
from pandas import read_csv
from utils import minutizer, preprocess_2


def lstm_model(stock: str,
               lookback: int = 24,
               epochs: int = 100,
               batch_size: int = 96,
               learning_rate: float = 0.001,
               dropout_rate: float = 0.1,
               ground_features: int = 5):
    # Import data
    data = preprocess_2(minutizer(read_csv('../data/Health-Care/'+stock+'.csv', index_col='timestamp',
                                           parse_dates=True), split=5), stock)
    # Transform data
    n, d = data.shape
    train_val_test_split = {'train': 0.7, 'val': 0.85, 'test': 1}
    data = data.values
    min_max_scalar = {i: (min(data[0: int(n*train_val_test_split['val']), i]),
                          max(data[0: int(n*train_val_test_split['val']), i])) for i in range(1, d)}
    for i in range(1, d):
        data[:, i] = (data[:, i] - min_max_scalar[i][0])/(min_max_scalar[i][1] - min_max_scalar[i][0])

    X = np.zeros((n - lookback, lookback, d))
    Y = np.zeros((n - lookback, int(d/ground_features)))
    for i in range(X.shape[0]):
        for j in range(d):
            X[i, :, j] = data[i:(i+lookback), j]
            if j < int(d/ground_features):
                Y[i, j] = data[lookback + i, j * ground_features]

    X_train = X[0: int(n * train_val_test_split['train'])]
    y_train = Y[0: int(n * train_val_test_split['train'])]

    X_val = X[int(n*train_val_test_split['train']): int(n*train_val_test_split['val'])]
    y_val = Y[int(n*train_val_test_split['train']): int(n*train_val_test_split['val'])]

    # This is just added here for simplicity.
    X_test = X[int(n * train_val_test_split['val']): int(n * train_val_test_split['test'])]
    y_test = Y[int(n * train_val_test_split['val']): int(n * train_val_test_split['test'])]

    # Build the RNN
    model = Sequential()

    # Adding layers. LSTM(units) --> Dropout(p)
    model.add(LSTM(units=40, return_sequences=True, use_bias=True, input_shape=(X_train.shape[1], d)))
    model.add(Dropout(dropout_rate))

    model.add(LSTM(units=20, return_sequences=True, use_bias=False))
    model.add(Dropout(dropout_rate))

    model.add(LSTM(units=10, return_sequences=True, use_bias=False))
    model.add(Dropout(dropout_rate))

    model.add(LSTM(units=5, use_bias=False))
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(units=int(d/ground_features), activation='linear', use_bias=True))

    # Optimizer
    adam_opt = optimizers.adam(lr=learning_rate)

    # Compile
    model.compile(optimizer=adam_opt, loss='mean_squared_error')

    # Fit
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # Validate
    predicted_stock_returns = model.predict(X_val)

    # Save predictions on test and validation set
    pd.DataFrame(predicted_stock_returns).to_csv('../output/RNN_results/predictions/val_files/'+stock+
                                                 '_val_predictions.csv', index=False)
    pd.DataFrame(y_val).to_csv('../output/RNN_results/predictions/val_files/'+stock+'_val_real.csv', index=False)
    pd.DataFrame(model.predict(X_test)).to_csv('../output/RNN_results/predictions/test_files/' + stock +
                                               '_test_predictions.csv', index=False)
    pd.DataFrame(y_test).to_csv('../output/RNN_results/predictions/test_files/' + stock + '_test_real.csv', index=False)