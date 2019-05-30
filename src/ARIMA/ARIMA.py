from statsmodels.tsa.statespace.varmax import VARMAX
from utils import combine_ts, minutizer
import matplotlib.pyplot as plt

# Get data
tickers = ['AAP', 'MRK', 'NRG', 'ORLY']
data = minutizer(combine_ts(tickers))
n, _ = data.shape

# Split data
train_val_test_split = {'train': 0.5, 'val': 0.65, 'test': 1}
train_data = data[0: int(n*train_val_test_split['train'])]
val_data = data[int(n*train_val_test_split['train']): int(n*train_val_test_split['val'])]

# split data in X and Y
y_list = [ticker+'_close' for ticker in tickers]
# Train
endog_y = train_data[y_list]
exog_x = train_data.drop(columns=y_list)
# Validate
endog_y_val = val_data[y_list]
exog_x_val = val_data.drop(columns=y_list)

# Fit model
model = VARMAX(endog=endog_y.values, exog=exog_x.values, order=(1, 0))
model_fit = model.fit(disp=False)

# make prediction
predictions = model_fit.forecast(steps=exog_x_val.shape[0], exog=exog_x_val.values)

# Evaluate
for i, ticker in enumerate(tickers):
    MSE = sum((predictions[:, i] - endog_y_val.values[:, i])**2)/endog_y_val.shape[0]
    print(ticker+' MSE:', MSE)
