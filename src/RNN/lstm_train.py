from RNN.lstm_single import lstm_model
tickers = 'ABT'#, 'ABT']#, 'ALGN', 'SYK', 'UNH', 'VAR', 'WAT', 'ZTS']
test = lstm_model(tickers, epochs=1)
