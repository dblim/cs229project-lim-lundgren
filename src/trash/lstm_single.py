from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import optimizers
import numpy as np
import pandas as pd
from pandas import read_csv
from lstm_utils import minutizer, preprocess_2_single


def lstm_model(stock: str,
               lookback: int = 24,
               epochs: int = 100,
               batch_size: int = 96,
               learning_rate: float = 0.0001,
               dropout_rate: float = 0.1,
               ground_features: int = 5):
    # Import data
    data = preprocess_2_single(minutizer(read_csv('../data/sectors/Information-Technology/'+stock+'.csv',
                                                  index_col='timestamp', parse_dates=True), split=5), stock)
    # Transform data
    n, d = data.shape
    train_val_test_split = {'train': 0.7, 'val': 0.85, 'test': 1}
    data = data.values
    min_max_scalar = {i: (min(data[0: int(n*train_val_test_split['val']), i]),
                          max(data[0: int(n*train_val_test_split['val']), i])) for i in range(2, d-1)}
    for i in range(2, d-1):
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

    # Build the LSTM
    model = Sequential()

    # Adding layers. LSTM(units) --> Dropout(p)
    model.add(LSTM(units=ground_features, return_sequences=True, use_bias=True, input_shape=(X_train.shape[1], d)))
    model.add(Dropout(dropout_rate))

    #model.add(LSTM(units=3, return_sequences=True, use_bias=False))  # Could be removed
    #model.add(Dropout(dropout_rate))

    model.add(LSTM(units=1, use_bias=False))
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(units=1, activation='linear', use_bias=True))

    # Optimizer
    adam_opt = optimizers.adam(lr=learning_rate)

    # Compile
    model.compile(optimizer=adam_opt, loss='mean_squared_error')
    print(model.summary())

    # Fit
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    # Validate
    predicted_stock_returns = model.predict(X_val)

    # Save predictions on test and validation set
    pd.DataFrame(predicted_stock_returns).to_csv('../output/RNN_results/predictions/val_files/'+stock+
                                                 '_val_predictions.csv', index=False)
    pd.DataFrame(y_val).to_csv('../output/RNN_results/predictions/val_files/'+stock+'_val_real.csv', index=False)
    pd.DataFrame(model.predict(X_test)).to_csv('../output/RNN_results/predictions/test_files/' + stock +
                                               '_test_predictions.csv', index=False)
    pd.DataFrame(y_test).to_csv('../output/RNN_results/predictions/test_files/' + stock + '_test_real.csv', index=False)


    #print(history.history['loss'])
    predcted_returns = predicted_stock_returns.copy()
    actual_returns = y_val.copy()
    #
    MSE = sum((predcted_returns - actual_returns) ** 2) / y_val.shape[0]
    dummy_mse = sum(actual_returns ** 2) / (y_val.shape[0])
    print('=========', stock, '=========')
    print('Dummy MSE:', dummy_mse)
    print('MSE:', MSE)
    print('--')
    pred_zero_one = predicted_stock_returns.copy()
    pred_zero_one[pred_zero_one > 0] = 1
    pred_zero_one[pred_zero_one < 0] = 0
    print('Predicted ones:', np.mean(pred_zero_one))
    real_zero_one = y_val.copy()
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
    accuracy = (TP + TN) / (TP + TN + FP + FN)
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
    print('Dummy Sharpe Ration:', np.mean(actual_returns) / np.std(actual_returns))
    print('Strategy return:', (strategy_return - 1) * 100)
    print('Strategy standard deviation: ', np.std(obvious_strategy))
    print('Strategy Sharpe Ration:', np.mean(obvious_strategy) / np.std(obvious_strategy))
    print('Return Correlation:', np.corrcoef(predicted_stock_returns.T, y_val.T)[0][1])

