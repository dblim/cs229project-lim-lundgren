from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import keras.backend as K
from keras import optimizers
import numpy as np
import pandas as pd
import random
from lstm_utils import minutizer, combine_ts, preprocess_2_multi, customized_loss

rand_tuning : bool = True
det_tuning : bool = False

data = pd.read_csv('../data/preprocessed_time_series_data.csv')
data = data.drop(columns=['Unnamed: 0'])


def lstm_model_mse(lstm_units :int, lookback : int , learning_rate : float   stocks: list,
               epochs: int = 40,
                batch_size : int =  96,
                dropout_rate : 0.1
               ground_features: int = 4):
    amount_of_stocks = 10

    # Transform data according to lstm_partial_ts
    n, d = data.shape
    train_val_test_split = {'train': 0.7, 'val': 0.85, 'test': 1}

    X = np.zeros((amount_of_stocks, n - lookback, lookback, ground_features))
    Y = np.zeros((amount_of_stocks, n - lookback, ground_features))
    for i in range(amount_of_stocks):
        for j in range(X.shape[0]):
            for k in range(ground_features):
                idx = i * ground_features + k
                X[i, j, :, k] = data.values[i: (i + lookback), ]

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
    model.add(LSTM(units=10, return_sequences=True, use_bias=True, input_shape=(X_train.shape[1], d)))
    model.add(Dropout(dropout_rate))

    model.add(LSTM(units=int(d/ground_features), use_bias=False))
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(units=int(d/ground_features), activation='linear', use_bias=True))

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

        predcted_returns = predicted_stock_returns[:, i].copy()
        actual_returns = y_val[:, i].copy()
        #
        MSE = sum((predcted_returns - actual_returns) ** 2) / y_val.shape[0]

    avg_mse = np.array(all_mse)
    return np.mean(avg_mse)

tickers = ['ACN', 'AMAT' ,    'CDNS', 'IBM', 'INTU', 'LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']

if rand_tuning is True:

    random.seed()

    # Search lstm size between 10 and 100
    # Search lookback period between 10 and 40
    # Search dropout rate between 0.1 and 0.5

    lstm_units_list = []
    lookback_list = []
    dropout_list = []
    avg_mse_list = []

    # Choose 2 random pairs of numbers for lstm units, lookback, dropout rate
    num_trials = 15
    for k in range(num_trials):
        lstm_units, lookback, dropout_rate = random.randint(10, 60), random.randint(10, 40), random.uniform(0.1, 0.5)
        avg_mse = lstm_model_mse(lstm_units, lookback, dropout_rate, tickers)
        print('Average MSE:', avg_mse)
        print('Number of LSTM units:', lstm_units)
        print('Lookback period:', lookback)
        print('Dropout rate:', dropout_rate)
        lstm_units_list.append(lstm_units)
        lookback_list.append(lookback)
        dropout_list.append(dropout_rate)
        avg_mse_list.append(avg_mse)

    # Save MSE computations to pandas datqframe
    df = pd.DataFrame( list(zip(lstm_units_list, lookback_list, dropout_list, avg_mse_list)), \
                        columns = ['Number of LSTM units', 'Lookback period', 'Dropout rate', 'Average MSE' ])
    random_integer = random.randint(1,100)
    pd.DataFrame(df).to_csv('../output/LSTM_tuning/rand_tuning_'  + str(random_integer) + '_epochs_' +  str(2) +  '.csv', index=False)

if det_tuning is True:
    periods = [4,8,12,16,20,24,28,32,36,40]

    avg_mse_list = []
    for lookback in periods:
        avg_mse = lstm_model_mse( lookback, tickers)
        print('Lookback period:', lookback)
        print('Average MSE:', avg_mse)
        avg_mse_list.append(avg_mse)
    # Save MSE computations to pandas dataframe
    df = pd.DataFrame(list(zip(periods, avg_mse_list)), \
                      columns=['Lookback period', 'Average MSE'])
    pd.DataFrame(df).to_csv('../output/LSTM_tuning/det_tuning_' + str(4_40) + '_epochs_' + str(40) + '.csv',\
                            index=False)




