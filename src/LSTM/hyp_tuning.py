from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Concatenate, Input
import keras.backend as K
from keras import optimizers
import numpy as np
import pandas as pd
import random
from lstm_utils import minutizer, combine_ts, preprocess_2_multi, customized_loss

rand_tuning : bool = False
det_tuning : bool = False

data = pd.read_csv('../data/preprocessed_time_series_data.csv')
data = data.drop(columns=['Unnamed: 0'])


def lstm_model_mse(lookback: int,  # HP,
                learning_rate: float,  # HP
                output_dim_individual_layer: int,  # HP
                stocks: list,
                epochs: int = 100,  # HP
                batch_size: int = 96,  # HP
                output_dim_combined_layer: int = 10,  # = amount of stocks
                dropout_rate: float = 0.1,  # HP
                ground_features: int = 4,  # this could be changed but let's keep it this way..
                percentile: int = 10):  # just for checking
    amount_of_stocks = 10

    # Transform data
    n, d = data.shape
    train_val_test_split = {'train': 0.7, 'val': 0.85, 'test': 1}
    X = np.zeros((amount_of_stocks, n - lookback, lookback, ground_features))
    Y = np.zeros((n - lookback, amount_of_stocks))
    for i in range(amount_of_stocks):
        for j in range(X.shape[1]):
            for k in range(ground_features):
                idx = i * ground_features + k
                X[i, j, :, k] = data.values[j: (j + lookback), idx]
            Y[j, i] = data.values[j + lookback, i * ground_features]

    X_train = X[:, 0: int(n * train_val_test_split['train'])]
    y_train = Y[0: int(n * train_val_test_split['train'])]

    X_val = X[:, int(n*train_val_test_split['train']): int(n*train_val_test_split['val'])]
    y_val = Y[int(n*train_val_test_split['train']): int(n*train_val_test_split['val'])]

    X_test = X[:, int(n * train_val_test_split['val']): int(n * train_val_test_split['test'])]
    y_test = Y[int(n * train_val_test_split['val']): int(n * train_val_test_split['test'])]

    # Create the network

    ts1 = Input((X_train.shape[2], X_train.shape[3]))
    LSTM1 = LSTM(output_dim_individual_layer, return_sequences=True)(ts1)
    drop1 = Dropout(dropout_rate)(LSTM1)

    ts2 = Input((X_train.shape[2], X_train.shape[3]))
    LSTM2 = LSTM(output_dim_individual_layer, return_sequences=True)(ts2)
    drop2 = Dropout(dropout_rate)(LSTM2)

    ts3 = Input((X_train.shape[2], X_train.shape[3]))
    LSTM3 = LSTM(output_dim_individual_layer, return_sequences=True)(ts3)
    drop3 = Dropout(dropout_rate)(LSTM3)

    ts4 = Input((X_train.shape[2], X_train.shape[3]))
    LSTM4 = LSTM(output_dim_individual_layer, return_sequences=True)(ts4)
    drop4 = Dropout(dropout_rate)(LSTM4)

    ts5 = Input((X_train.shape[2], X_train.shape[3]))
    LSTM5 = LSTM(output_dim_individual_layer, return_sequences=True)(ts5)
    drop5 = Dropout(dropout_rate)(LSTM5)

    ts6 = Input((X_train.shape[2], X_train.shape[3]))
    LSTM6 = LSTM(output_dim_individual_layer, return_sequences=True)(ts6)
    drop6 = Dropout(dropout_rate)(LSTM6)

    ts7 = Input((X_train.shape[2], X_train.shape[3]))
    LSTM7 = LSTM(output_dim_individual_layer, return_sequences=True)(ts7)
    drop7 = Dropout(dropout_rate)(LSTM7)

    ts8 = Input((X_train.shape[2], X_train.shape[3]))
    LSTM8 = LSTM(output_dim_individual_layer, return_sequences=True)(ts8)
    drop8 = Dropout(dropout_rate)(LSTM8)

    ts9 = Input((X_train.shape[2], X_train.shape[3]))
    LSTM9 = LSTM(output_dim_individual_layer, return_sequences=True)(ts9)
    drop9 = Dropout(dropout_rate)(LSTM9)

    ts10 = Input((X_train.shape[2], X_train.shape[3]))
    LSTM10 = LSTM(output_dim_individual_layer, return_sequences=True)(ts10)
    drop10 = Dropout(dropout_rate)(LSTM10)

    merged = Concatenate()([drop1, drop2, drop3, drop4, drop5, drop6, drop7, drop8, drop9, drop10])
    full_LSTM = LSTM(output_dim_combined_layer)(merged)
    full_drop = Dropout(dropout_rate)(full_LSTM)
    output_layer = Dense(output_dim_combined_layer)(full_drop)

    full_model = Model(inputs=[ts1, ts2, ts3, ts4, ts5, ts6, ts7, ts8, ts9, ts10], outputs=output_layer)

    adam_opt = optimizers.adam(lr=learning_rate)

    # Compile
    full_model.compile(optimizer=adam_opt, loss=customized_loss)

    # Fit
    history = full_model.fit([X_train[i] for i in range(amount_of_stocks)],
                             y_train, epochs=epochs, batch_size=batch_size,
                             validation_data=([X_val[i] for i in range(amount_of_stocks)], y_val))

    predicted_stock_returns_val = full_model.predict([X_val[i] for i in range(amount_of_stocks)])

    all_mse = []
    for i, ticker in enumerate(stocks):
        predcted_returns = predicted_stock_returns_val[:, i].copy()
        actual_returns = y_val[:, i].copy()
        #
        MSE = sum((predcted_returns - actual_returns) ** 2) / y_val.shape[0]
        all_mse.append(MSE)
    avg_mse = np.array(all_mse)
    return np.mean(avg_mse)

tickers = ['ACN', 'AMAT' ,    'CDNS', 'IBM', 'INTU', 'LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']

if rand_tuning is True:

    random.seed()

    # Search lstm size between 10 and 100
    # Search lookback period between 10 and 40
    # Search dropout rate between 0.1 and 0.5


    lookback_list = []
    learning_rate_list = []
    output_dim_individual_layer_list = []
    avg_mse_list = []

    # Choose 2 random pairs of numbers for lstm units, lookback, dropout rate
    num_trials = 4
    for k in range(num_trials):
        lookback  = random.randint(10, 40)
        learning_rate =    np.random.choice([0.1, 0.01, 0.001, 0.0001, 0.5, 0.05, 0.005, 0.0005])
        output_dim_individual_layer = random.randint( 1, 40)
        avg_mse = lstm_model_mse( lookback, learning_rate, output_dim_individual_layer, tickers)

        print('Number of LSTM units:', output_dim_individual_layer)
        print('Lookback period:', lookback)
        print('Learning rate:', learning_rate)
        print('Average MSE:', avg_mse)

        output_dim_individual_layer_list.append(output_dim_individual_layer)
        lookback_list.append(lookback)
        learning_rate_list.append(learning_rate)
        avg_mse_list.append(avg_mse)

    # Save MSE computations to pandas datqframe
    df = pd.DataFrame( list(zip(output_dim_individual_layer_list, lookback_list, learning_rate_list, avg_mse_list)), \
                        columns = ['Number of LSTM units', 'Lookback period', 'Learning rate', 'Average MSE' ])
    random_integer = random.randint(1,100)
    pd.DataFrame(df).to_csv('../output/LSTM_tuning/random_samples/rand_tuning_' + \
                             str(random_integer) + '_epochs_' +  str(50) +  '.csv', index=False)

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


lookback = 24
learning_rate = 1e-4
output_dim_individual_layer = 1
print( lstm_model_mse( lookback, learning_rate, output_dim_individual_layer, tickers))


