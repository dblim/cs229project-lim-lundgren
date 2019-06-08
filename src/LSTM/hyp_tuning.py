from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import keras.backend as K
from keras import optimizers
import numpy as np
import pandas as pd
import random
from lstm_utils import minutizer, combine_ts, preprocess_2_multi, customized_loss

random : bool = False
deterministic : bool = True

data = pd.read_csv('../data/preprocessed_time_series_data.csv')
data = data.drop(columns=['Unnamed: 0'])


def lstm_model_mse(lstm_units :int, lookback : int ,  stocks: list,
               epochs: int = 40,
                batch_size : int =  96,
               learning_rate: float = 0.0001,
                dropout_rate : float = 0.1,
               ground_features: int = 4):

    # Transform data (This transformation is according to lstm_multi and NOT change)
    n, d = data.shape
    train_val_test_split = {'train': 0.7, 'val': 0.85, 'test': 1}

    X = np.zeros((n - lookback, lookback, d))
    Y = np.zeros((n - lookback, int(d / ground_features)))
    for i in range(X.shape[0]):
        for j in range(d):
            X[i, :, j] = data.iloc[i:(i + lookback), j]
            if j < int(d / ground_features):
                Y[i, j] = data.iloc[lookback + i, j * ground_features]

    X_train = X[0: int(n * train_val_test_split['train'])]
    y_train = Y[0: int(n * train_val_test_split['train'])]

    X_val = X[int(n * train_val_test_split['train']): int(n * train_val_test_split['val'])]
    y_val = Y[int(n * train_val_test_split['train']): int(n * train_val_test_split['val'])]

    X_test = X[int(n * train_val_test_split['val']): int(n * train_val_test_split['test'])]
    y_test = Y[int(n * train_val_test_split['val']): int(n * train_val_test_split['test'])]

    # Hyperparameter printing

    # Adding layers. LSTM(n) --> Dropout(p)
    model.add(LSTM(units=d, return_sequences=True, use_bias=True, input_shape=(X_train.shape[1], d)))
    model.add(Dropout(dropout_rate))

    model.add(LSTM(units=int(d / ground_features), use_bias=False))
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(units=int(d / ground_features), activation='linear', use_bias=True))

    # Optimizer
    adam_opt = optimizers.adam(lr=learning_rate)

    # Compile
    model.compile(optimizer=adam_opt, loss=customized_loss)

    # Fit
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    # Validate
    predicted_stock_returns = model.predict(X_val)

    all_mse = []




    for i, ticker in enumerate(stocks):
        predcted_returns = predicted_stock_returns[:, i].copy()
        actual_returns = y_val[:, i].copy()
        #
        MSE = sum((predcted_returns - actual_returns) ** 2) / y_val.shape[0]
        all_mse.append(MSE)

    avg_mse = np.array(all_mse)
    return np.mean(avg_mse)

tickers = ['ACN', 'AMAT' ] #    'CDNS', 'IBM', 'INTU', 'LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']

if random is True:
    # Choose 2 random pairs of numbers for lstm units, batch size

    random.seed()

    # Search lstm size between 10 and 100
    # Search batch size between 50 and 150.
    # Search dropout rate between 0.2

    lstm_units_list = []
    lookback_list = []
    avg_mse_list = []

    for k in range(2):
        lstm_units, lookback = random.randint(10, 60), random.randint(10, 40)
        avg_mse = lstm_model_mse(lstm_units, lookback, tickers)
        print('Average MSE:', avg_mse)
        print('Number of LSTM units:', lstm_units)
        print('Lookback period:', lookback)
        lstm_units_list.append(lstm_units)
        lookback_list.append(lookback)
        avg_mse_list.append(avg_mse)


    # Save MSE computations to pandas dataframe
    df = pd.DataFrame( list(zip(lstm_units_list, lookback_list, avg_mse_list)), \
                        columns = ['Number of LSTM units', 'Lookback period', 'Average MSE' ])
    random_integer = random.randint(1,100)
    pd.DataFrame(df).to_csv('../output/LSTM_tuning/rand_tuning'  + str(random_integer) + '_epochs_' +  str(40) +  '.csv', index=False)

if deterministic is True:
    periods = [4,8,12]
    lstm_units = 25
    avg_mse_list = []
    for lookback in periods:
        avg_mse = lstm_model_mse(lstm_units, lookback, tickers)
        print('Lookback period:', lookback)
        print('Average MSE:', avg_mse)
        avg_mse_list.append(avg_mse)
    # Save MSE computations to pandas dataframe
    df = pd.DataFrame(list(zip(periods, avg_mse_list)), \
                      columns=['Lookback period', 'Average MSE'])
    pd.DataFrame(df).to_csv('../output/LSTM_tuning/det_tuning' + str(4_8_12) + '_epochs_' + str(2) + '.csv',\
                            index=False)




