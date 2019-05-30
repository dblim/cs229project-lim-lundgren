
from RNN.LSTM import lstm_model
#stocks = ['AAP', 'CRM']
stocks = 'AAP'
test = lstm_model(stocks, epochs=30)
