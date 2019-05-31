from RNN.R2N2 import lstm_model
tickers = ['AAP', 'MRK', 'NRG', 'ORLY']
test = lstm_model(tickers, epochs=50)
