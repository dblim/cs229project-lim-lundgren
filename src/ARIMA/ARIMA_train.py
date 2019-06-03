from ARIMA.ARIMA import varmax
tickers = ['ACN', 'AMAT', 'CDNS', 'IBM', 'INTU', 'LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']

for i in range(11):
    p = (i + 1) * 2
    test = varmax(tickers, p=p)