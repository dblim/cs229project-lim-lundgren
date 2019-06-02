from RNN.lstm_multi import lstm_model
tickers = ['AAP', 'MRK', 'NRG', 'ORLY']
test = lstm_model(tickers, epochs=100)
