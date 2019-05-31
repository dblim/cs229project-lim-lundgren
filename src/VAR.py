import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR, VARMAX
from utils import combine_ts_returns

# data
tickers = ['AAP', 'AES', 'AMD', 'BSX', 'CHD', 'CMG', 'CRM', 'EW', 'FOX', 'FOXA']
data = combine_ts_returns(tickers)

# Split data
n = np.shape(data)[0]
train_data = data[0 : int(0.8*n)]
val_data = data[int(0.8*n) : ]

# Train on returns
y_list = [t+'_returns' for t in tickers]
endog_y = train_data[y_list]
exog_x = train_data.drop(columns=y_list)


# Validate
endog_y_val = val_data[y_list]
exog_x_val = val_data.drop(columns=y_list)

# We choose a hyperparameter of p = 1

model = VAR(endog_y.values)
results = model.fit(1)
residuals = results.resid

results_values = results.fittedvalues
print(residuals[0])
print(results_values[0])
print(endog_y)













# Get residual on train data to train for RNN
#endog_y = endog_y[var_returns.index]
#endog_y_returns = endog_y[var_returns.columns]
#residual = endog_y_returns - var_returns
#print(residual)




#predictions_1 = model.fit(1).forecast(endog_y.values, steps=2)
#print(predictions_1)






# predictions = prediction_function(endog_y.values, 10)

# Prints MSE error for each p.
"""for p in range(len(predictions)):
    for i, ticker in  enumerate( tickers):
        MSE = sum((predictions[p-1][:, i] - endog_y_val.values[:, i])**2)/endog_y_val.shape[0]
        print("For p = {} and {}, ".format(p+1, ticker) + "MSE error is", MSE)"""

"""def prediction_function(values, p):
    model = VAR(values)
    return [ model.fit(i).forecast(values, steps=10) for i in range(1,p)]"""

