from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import zero_one_loss
from sklearn.preprocessing import MinMaxScaler
from utils import minutizer, combine_ts, preprocess_arima


def lstm_model(stocks: list,
               lookback: int = 12,
               epochs: int = 10):
    # Import data
    data, opens = preprocess_arima(minutizer(combine_ts(stocks), split=5), stocks)
    header = list(data)

    # Transform data
    n, d = data.shape
    train_val_test_split = {'train': 0.7, 'val': 0.85, 'test': 1}

    X = np.zeros((n - lookback, lookback, d))
    Y = np.zeros((n - lookback, int(d/3)))
    for i in range(X.shape[0]):
        for j in range(d):
            X[i, :, j] = data.iloc[i:(i+lookback), j]
            if j < int(d/3):
                Y[i, j] = data.iloc[lookback + i, j * 3]

    X_train = X[0: int(n * train_val_test_split['train'])]
    print(X_train.shape)
    y_train = Y[0: int(n * train_val_test_split['train'])]

    X_val = X[int(n*train_val_test_split['train']): int(n*train_val_test_split['val'])]
    y_val = Y[int(n*train_val_test_split['train']): int(n*train_val_test_split['val'])]

    opens_val = opens[int(n*train_val_test_split['train']): int(n*train_val_test_split['val'])]


    # Initialising the RNN
    model = Sequential()

    # Adding layers. LSTM(25) --> Dropout(0.2) x 4
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], d)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(units=int(d/3), activation='linear'))

    # Run
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to the Training set
    model.fit(X_train, y_train, epochs=epochs, batch_size=96)

    # Validate
    predicted_stock_price = model.predict(X_val)

    for i, ticker in enumerate(stocks):
        pred = (predicted_stock_price[:, i] + 1) * opens_val.values[:, i]
        real = (y_val[:, i] + 1) * opens_val.values[:, i]
        MSE = sum((pred - real) ** 2) / y_val.shape[0]
        print('=========', ticker, '=========')
        print(' MSE:', MSE)
        pred_zero_one = predicted_stock_price[:, i]
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

        plt.plot(real, color='red', label='Real ' + ticker + ' Stock Price')
        plt.plot(pred, color='blue', label='Predicted ' + ticker + ' Stock Price')
        plt.title(ticker + ' Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel(ticker + ' Stock Price')
        plt.legend()
        plt.savefig('../output/RNN_results/LSTM_new_test_' + ticker + '.png')
        plt.close()
