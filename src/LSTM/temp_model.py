#coding=utf-8

from __future__ import print_function

try:
    import numpy as np
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from keras.datasets import mnist
except:
    pass

try:
    from keras.layers.core import Dense, Dropout, Activation
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.utils import np_utils
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.layers import Dense, LSTM, Dropout
except:
    pass

try:
    import keras.backend as K
except:
    pass

try:
    from keras import optimizers
except:
    pass

try:
    import numpy as np
except:
    pass

try:
    import pandas as pd
except:
    pass

try:
    import random
except:
    pass

try:
    from lstm_utils import minutizer, combine_ts, preprocess_2_multi
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

lookback = 24
ground_features = 4
dropout_rate = float(0.1)
percentile = 10
learning_rate = float(0.0001)
stocks = ['ACN', 'AMAT',  'CDNS', 'IBM', 'INTU', 'LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']
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


def keras_fmin_fnct(space):

    lookback = 24
    ground_features = 4
    dropout_rate = float(0.1)
    percentile = 10
    learning_rate = float(0.0001)
    model = Sequential()
    # Adding layers. LSTM(n) --> Dropout(p)
    model.add(LSTM(units=space['units'], return_sequences=True, use_bias=True, input_shape=(lookback, ground_features)))
    model.add(Dropout(dropout_rate))

    model.add(LSTM(units=10, use_bias=False))
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(units=1, activation='linear', use_bias=True))

    # Optimizer
    adam_opt = optimizers.adam(lr=learning_rate)

    # Compile
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit
    result = model.fit(X_train, y_train, epochs=2, batch_size=96, validation_data=(X_val, y_val))

    #get the highest validation accuracy of the training epochs
    validation_acc = result.history['val_acc']
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

def get_space():
    return {
        'units': hp.choice('units', [20,30,40]),
    }
