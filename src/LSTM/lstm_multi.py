from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import optimizers
import numpy as np
import pandas as pd
from utils import minutizer, combine_ts, preprocess_2_multi


def lstm_model(stocks: list,
               lookback: int = 24,
               epochs: int = 100,
               batch_size: int = 96,
               learning_rate: float = 0.0001,
               dropout_rate: float = 0.1,
               ground_features: int = 5):
    # Import data
    data = minutizer(combine_ts(stocks), split=5)

    data = preprocess_2_multi(data, stocks)

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
    model.add(LSTM(units=d, return_sequences=True, use_bias=True, input_shape=(X_train.shape[1], d)))
    model.add(Dropout(dropout_rate))

    #model.add(LSTM(units=3, return_sequences=True, use_bias=False))
    #model.add(Dropout(dropout_rate))

    model.add(LSTM(units=int(d/ground_features), use_bias=False))
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(units=int(d/ground_features), activation='linear', use_bias=True))

    # Optimizer
    adam_opt = optimizers.adam(lr=learning_rate)

    # Compile
    model.compile(optimizer=adam_opt, loss='mean_squared_error')

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
        threshold = np.percentile(predcted_returns, 10)
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
        print('Correlation:', 5)