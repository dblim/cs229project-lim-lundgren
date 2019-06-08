import matplotlib.pyplot as plt
import pandas as pd

tickers = ['ACN', 'AMAT', 'CDNS', 'IBM', 'INTU', 'LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']
ticker = 'IBM'

for ticker in tickers:

    pred_path = '../output/LSTM_results/single_valid/'+ticker+'_val_predictions.csv'
    real_path = '../output/LSTM_results/single_valid/'+ticker+'_val_real.csv'

    pred = pd.read_csv(pred_path).values
    real = pd.read_csv(real_path).values

    plt.hist(real, bins=100, density=True)
    plt.hist(pred,  bins=100, density=True, alpha=0.75)
    plt.axis([-0.01, 0.01, 0, 1000])
    plt.show()