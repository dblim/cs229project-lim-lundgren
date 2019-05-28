from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import zero_one_loss


# Import data
path = '../data/top_stocks/CRM.csv'
data = pd.read_csv(path, index_col="timestamp", parse_dates=True)
print(data.head())


# Transform data
n, d = data.shape
train_val_test_split = {'train': 0.5, 'val': 0.625, 'test': 1}
sc = MinMaxScaler(feature_range=(0, 1))
data_set_scaled = sc.fit_transform(data)#['close'].values.reshape(n, 1))
lookback = 60
X_close = []
X_high = []
X_low = []
X_open = []
X_vol = []
Y = []
for i in range(lookback, len(data_set_scaled)):
    X_close.append(data_set_scaled[(i-lookback):i, 0])
    X_high.append(data_set_scaled[(i - lookback):i, 1])
    X_low.append(data_set_scaled[(i - lookback):i, 2])
    X_open.append(data_set_scaled[(i - lookback):i, 3])
    X_vol.append(data_set_scaled[(i - lookback):i, 4])
    Y.append(data_set_scaled[i, 0])
Y = np.array(Y)
#X = np.reshape(X, (X.shape[0], X.shape[1], 1))
X = np.stack((np.array(X_close), np.array(X_high), np.array(X_low), np.array(X_open), np.array(X_vol)), axis=2)
print(X.shape)

X_train = X[0: int(n * train_val_test_split['train'])]
y_train = Y[0: int(n * train_val_test_split['train'])]

X_val = X[int(n*train_val_test_split['train']): int(n*train_val_test_split['val'])]
y_val = Y[int(n*train_val_test_split['train']): int(n*train_val_test_split['val'])]


# Initialising the RNN
model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], d)))
model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units=50))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units=1))
# Compiling the RNN
model.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Validate
predicted_stock_price = model.predict(X_val)
#predicted_stock_price = sc.inverse_transform(predicted_stock_price)
#real_stock_price = sc.inverse_transform(y_val.reshape(len(y_val), 1))
predicted_stock_price = sc.inverse_transform(np.concatenate((predicted_stock_price, np.zeros((len(y_val), d-1))), axis=1))[:, 0]
real_stock_price = sc.inverse_transform(np.concatenate((y_val.reshape(len(y_val), 1), np.zeros((len(y_val), d-1))), axis=1))[:, 0]

# Plot
plt.plot(real_stock_price, color='red', label='Real CRM Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted CRM Stock Price')
plt.title('CRM Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('CRM Stock Price')
plt.legend()
plt.show()

# 0-1 loss

y_binary_hat = predicted_stock_price[1: predicted_stock_price.shape[0]]/real_stock_price[0: (predicted_stock_price.shape[0]-1)] -1
y_binary = real_stock_price[1: predicted_stock_price.shape[0]]/real_stock_price[0: (predicted_stock_price.shape[0]-1)] -1

y_binary[y_binary > 0] = 1
y_binary[y_binary < 0] = 0

y_binary_hat[y_binary_hat > 0] = 1
y_binary_hat[y_binary_hat < 0] = 0

print('Val set mean:', np.mean(y_binary))
print('Zero-One loss:', max(zero_one_loss(y_binary, y_binary_hat), 1-zero_one_loss(y_binary, y_binary_hat)))
