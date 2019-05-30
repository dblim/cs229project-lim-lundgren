from RNN.lstm_multi_new import lstm_model
tickers = ['AAP', 'MRK', 'NRG', 'ORLY']
test = lstm_model(tickers, epochs=30)
