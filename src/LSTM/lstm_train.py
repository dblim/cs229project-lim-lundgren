from trash.lstm_multi import lstm_model

"""
tickers = ['AAPL', 'ACN', 'ADBE', 'ADI', 'ADP', 'ADS', 'ADSK', 'AKAM', 'AMAT', 'AMD', 'ANSS', 'APH', 'AVGO', 'CDNS',
           'CRM', 'CSCO', 'CTSH', 'CTXS', 'DXC', 'FFIV', 'FIS', 'FISV', 'FLIR', 'GLW', 'GPN', 'HPE', 'HPQ', 'IBM',
           'INTC', 'INTU', 'IT', 'JNPR', 'KLAC', 'LRCX', 'MA', 'MCHP', 'MSFT', 'MSI', 'MU', 'NTAP', 'NVDA', 'ORCL',
           'PAYX', 'PYPL', 'QCOM', 'QRVO', 'RHT', 'SNPS', 'STX', 'SWKS', 'SYMC', 'TEL', 'TSS', 'TXN', 'V', 'VRSN',
           'WDC', 'WU', 'XLNX', 'XRX']
"""
tickers = ['XLNX', 'WDC']
use = []

lstm_model(tickers)

"""
pred = read_csv('../output/RNN_results/predictions/val_files/'+ticker+'_val_predictions.csv')
real = read_csv('../output/RNN_results/predictions/val_files/'+ticker+'_val_real.csv')
co = np.corrcoef(pred.values.T, real.values.T)
print(ticker+'_correlation:', co[0][1])
ones_real = real.values.copy()
ones_real[ones_real > 0] = 1
ones_real[ones_real < 0] = 0
ones_pred = pred.values.copy()
ones_pred[ones_pred > 0] = 1
ones_pred[ones_pred < 0] = 0
treshold = np.percentile(pred.values, 10)
invest = 1
strategy = 1
for i in range(pred.shape[0]):
    invest *= (1 + real.values[i])
    if real.values[i] > treshold:
        strategy *= (1 + real.values[i])
print('Strategy ret:', strategy)
print('Invest:', invest)
"""


# use = ['ACN', 'AMAT', 'CDNS', 'IBM', 'INTU', 'LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']
