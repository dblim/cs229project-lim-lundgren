from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import keras.backend as K
from keras import optimizers, losses
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from utils import minutizer, combine_ts, preprocess_2_multi


def preprocess_2_multi(data, tickers: list, ground_features: int = 5, new_features: int = 4):
    n, d = data.shape
    new_d = int(d/ground_features)
    new_data = np.zeros((n, new_d * new_features))
    open_prices = np.zeros((n, new_d))
    for i in range(new_d):
        new_data[:, new_features * i] = \
            data.iloc[:, ground_features * i]/data.iloc[:, ground_features * i + 3] - 1  # Returns
        new_data[:, new_features * i + 1] = \
            data.iloc[:, ground_features * i + 1] - data.iloc[:, ground_features * i + 2]  # Spread
        new_data[:, new_features * i + 2] = \
            (data.iloc[:, ground_features * i + 4] - np.min(data.iloc[:, ground_features * i + 4]))/ \
            (np.max(data.iloc[:, ground_features * i + 4]) - np.min(data.iloc[:, ground_features * i + 4]))  # Volume
        new_data[:, new_features * i + 3] = \
            (data.iloc[:, ground_features * i + 3] - np.min(data.iloc[:, ground_features * i + 3]))/ \
            (np.max(data.iloc[:, ground_features * i + 3]) - np.min(data.iloc[:, ground_features * i + 3]))  # open prize
        #new_data[:, new_features * i + 4] = \
        #    np.sin(2 * np.pi * new_data[:, new_features * i + 3]/np.max(new_data[:, new_features * i + 3]))  # Sin
        open_prices[:, i] = data.iloc[:, ground_features * i + 3]
    header_data = []
    header_open = []
    for ticker in tickers:
        header_data.append(ticker + '_returns')
        header_data.append(ticker + '_spread')
        header_data.append(ticker + '_volume')  # Normalized
        header_data.append(ticker + '_normalized_open')
        #header_data.append(ticker + '_sin_returns')
        header_open.append(ticker + '_open')
    return pd.DataFrame(new_data, columns=header_data), pd.DataFrame(open_prices, columns=header_open)


def minutizer(data, split: int = 5, ground_features: int = 5):
    n, d = data.shape
    new_data = pd.DataFrame(np.zeros((int(n/split) - 1, d)), columns=list(data))
    for i in range(int(n/split) - 1):
        for j in range(int(d/ground_features)):
            # Close
            new_data.iloc[i, j * ground_features] = data.iloc[split * (i + 1), j * ground_features]
            # High
            new_data.iloc[i, j * ground_features + 1] = max([data.iloc[split*i+k, j * ground_features + 1]
                                                             for k in range(split)])
            # Low
            new_data.iloc[i, j * ground_features + 2] = min([data.iloc[split * i + k, j * ground_features + 2]
                                                             for k in range(split)])
            # Open
            new_data.iloc[i, j * ground_features + 3] = data.iloc[split*i, j * ground_features + 3]
            # Volume
            new_data.iloc[i, j * ground_features + 4] = np.sum(data.iloc[i*split:(i+1)*split, j * ground_features + 4])
    return new_data


def combine_ts(tickers: list):
    stock0 = tickers[0]
    path = '../data/sectors/Information Technology/'+stock0+'.csv'
    data = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    renamer = {'close': stock0+'_close', 'high': stock0+'_high', 'low': stock0+'_low',
               'open': stock0+'_open', 'volume': stock0+'_volume', }
    data = data.rename(columns=renamer)
    tickers.remove(tickers[0])
    for str in tickers:
        path = '../data/sectors/Information Technology/'+str+'.csv'
        new_data = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        renamer = {'close': str+'_close', 'high': str+'_high', 'low': str+'_low',
                   'open': str+'_open', 'volume': str+'_volume', }
        new_data = new_data.rename(columns=renamer)

        data = pd.concat([data, new_data], axis=1, sort=True)
    tickers.insert(0, stock0)
    return data.interpolate()[1:data.shape[0]]


def customized_loss(y_pred, y_true):
    num = K.sum(K.square(y_pred - y_true), axis=-1)
    y_true_sign = y_true > 0
    y_pred_sign = y_pred > 0
    logicals = K.equal(y_true_sign, y_pred_sign)
    logicals_0_1 = K.cast(logicals, 'float32')
    den = K.sum(logicals_0_1, axis=-1)
    return num/(1 + den)


def lstm_model(stocks: list,
               lookback: int = 24,
               epochs: int = 50,
               batch_size: int = 96 ,
               learning_rate: float = 0.0002,
               dropout_rate: float = 0.1,
               ground_features: int = 4,
               percentile: int = 10):
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

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss loss')
    plt.legend()
    plt.savefig('losstjalala')
    plt.close()

    for i, ticker in enumerate(stocks):
        predcted_returns = predicted_stock_returns[:, i].copy()
        actual_returns = y_val[:, i].copy()
        #
        MSE = sum((predcted_returns - actual_returns) ** 2) / y_val.shape[0]
        dummy_mse = sum(actual_returns**2)/(y_val.shape[0])
        print('=========', ticker, '=========')
        print('Dummy MSE:', dummy_mse)
        print('MSE:', MSE)
        print('--')
        pred_zero_one = predicted_stock_returns[:, i]
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
        accuracy = (TP + TN)/(TP + TN + FP + FN)
        print('Accuracy:', max(accuracy, 1 - accuracy))
        print('--')
        dummy_return = 1
        strategy_return = 1
        threshold = np.percentile(predcted_returns, percentile)
        obvious_strategy = actual_returns[predcted_returns > threshold]
        for j in range(pred_zero_one.shape[0]):
            dummy_return *= (1 + actual_returns[j])
            if predcted_returns[j] > threshold:
                strategy_return *= (1 + actual_returns[j])
        print('Dummy return:', (dummy_return - 1) * 100)
        print('Dummy standard deviation: ', np.std(actual_returns))
        print('Dummy Sharpe Ration:', np.mean(actual_returns)/np.std(actual_returns))
        print('Strategy return:', (strategy_return - 1) * 100)
        print('Strategy standard deviation: ', np.std(obvious_strategy))
        print('Strategy Sharpe Ration:', np.mean(obvious_strategy) / np.std(obvious_strategy))
        print('Correlation:', np.corrcoef(predcted_returns.T, actual_returns.T)[0][1])


tickers = ['ACN', 'AMAT', 'CDNS', 'IBM', 'INTU', 'LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']
lstm_model(tickers)
