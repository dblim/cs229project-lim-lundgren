#from RNN.lstm_single import lstm_model
from pandas import read_csv
import numpy as np

tickers = ['AAPL', 'ACN', 'ADBE', 'ADI', 'ADP', 'ADS', 'ADSK', 'AKAM', 'AMAT', 'AMD', 'ANSS', 'APH', 'AVGO', 'CDNS',
           'CRM', 'CSCO', 'CTSH', 'CTXS', 'DXC', 'FFIV', 'FIS', 'FISV', 'FLIR', 'GLW', 'GPN', 'HPE', 'HPQ', 'IBM',
           'INTC', 'INTU', 'IT', 'JNPR', 'KLAC', 'LRCX', 'MA', 'MCHP', 'MSFT', 'MSI', 'MU', 'NTAP', 'NVDA', 'ORCL',
           'PAYX', 'PYPL', 'QCOM', 'QRVO', 'RHT', 'SNPS', 'STX', 'SWKS', 'SYMC', 'TEL', 'TSS', 'TXN', 'V', 'VRSN',
           'WDC', 'WU', 'XLNX', 'XRX']
use = []
for ticker in tickers:
    pred = read_csv('../output/RNN_results/predictions/val_files/'+ticker+'_val_predictions.csv')
    real = read_csv('../output/RNN_results/predictions/val_files/' + ticker + '_val_real.csv')
    print('===', ticker, '===')
    print('MSE:', sum((pred.values - real.values)**2)/real.shape[0])
    print('Corr:', np.corrcoef(pred.values.T, real.values.T)[0, 1])
    if np.corrcoef(pred.values.T, real.values.T)[0, 1] > 0.07:
        use.append(ticker)
print(use)

# use = ['ACN', 'AMAT', 'CDNS', 'IBM', 'INTU', 'LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']
