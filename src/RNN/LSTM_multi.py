from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import zero_one_loss
from utils import minutizer, combine_ts


def lstm_model(stocks: list,
               lookback: int = 12,
               epochs: int = 10):
    # Import data
    data = minutizer(combine_ts(stocks), split=5)
    header = list(data)

    # Transform data
    n, d = data.shape
    train_val_test_split = {'train': 0.7, 'val': 0.85, 'test': 1}

    y_data = np.zeros((n, int(d/5)))
    for i in range(int(d/5)):
        y_data[:, i] = data.values[:, 5 * i]
        data = data.drop(columns=header[3 + 5 * i])  # 3 corresponds to open price, which is removed.
    sc_y = MinMaxScaler(feature_range=(0, 1))
    sc_x = MinMaxScaler(feature_range=(0, 1))
    data_set_scaled_y = sc_y.fit_transform(y_data)
    data_set_scaled_x = sc_x.fit_transform(data)

    X = np.zeros((n - lookback, lookback, d - int(d/5)))
    Y = np.zeros((n - lookback, int(d/5)))
    for i in range(X.shape[0]):
        for j in range(d-int(d/5)):
            X[i, :, j] = data_set_scaled_x[i:(i+lookback), j]
        Y[i, :] = data_set_scaled_y[i, :]

    X_train = X[0: int(n * train_val_test_split['train'])]
    y_train = Y[0: int(n * train_val_test_split['train'])]

    X_val = X[int(n*train_val_test_split['train']): int(n*train_val_test_split['val'])]
    y_val = Y[int(n*train_val_test_split['train']): int(n*train_val_test_split['val'])]


    # Initialising the RNN
    model = Sequential()

    # Adding layers. LSTM(25) --> Dropout(0.2) x 4
    model.add(LSTM(units=25, return_sequences=True, input_shape=(X_train.shape[1], d - int(d/5))))
    model.add(Dropout(0.2))

    model.add(LSTM(units=25, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=25, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=25))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(units=int(d/5)))

    # Run
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to the Training set
    model.fit(X_train, y_train, epochs=epochs, batch_size=32)

    # Validate
    predicted_stock_price = sc_y.inverse_transform(model.predict(X_val))
    real_stock_price = sc_y.inverse_transform(y_val)

    # Plot
    for i in range(int(d/5)):
        predict = predicted_stock_price[:, i]
        real = real_stock_price[:, i]
        stock = stocks[i]
        plt.plot(real, color='red', label='Real '+stock+' Stock Price')
        plt.plot(predict, color='blue', label='Predicted '+stock+' Stock Price')
        plt.title(stock+' Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel(stock+' Stock Price')
        plt.legend()
        plt.savefig('../output/RNN_results/LSTM_'+stock+'.png')
        plt.close()

        np.save('../output/RNN_results/LSTM_'+stock+'_predictions', predict)

        # 0-1 loss and mean
        y_binary_hat = predict[1: predict.shape[0]]/real[0: (predict.shape[0]-1)] - 1
        y_binary = real[1: predict.shape[0]]/real[0: (predict.shape[0]-1)] - 1

        y_binary[y_binary > 0] = 1
        y_binary[y_binary < 0] = 0

        y_binary_hat[y_binary_hat > 0] = 1
        y_binary_hat[y_binary_hat < 0] = 0

        print(stock+' Val set mean:', np.mean(y_binary))
        print(stock+' Zero-One loss:', min(zero_one_loss(y_binary, y_binary_hat), 1-zero_one_loss(y_binary, y_binary_hat)))
        print(stock+' Mean squared error:', sum((real-predict)**2)/predict.shape[0])
