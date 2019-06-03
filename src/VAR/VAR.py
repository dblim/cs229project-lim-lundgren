from utils import combine_ts_returns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

"""In this script, we will get the training residuals for VAR. These will then be trained on using RNN/LSTM"""

# data
tickers = ['AAP', 'AMD']
data = combine_ts_returns(tickers)

# Split data
n = np.shape(data)[0]
train_data = data[0 : int(0.7*n)]
val_data = data[int(0.7*n) : int(0.85*n)]
test_data = data[ int(0.85*n) : ]

# Train on returns
y_list = [t+'_returns' for t in tickers]
endog_y = train_data[y_list]
exog_x = train_data.drop(columns=y_list)

# Validate
endog_y_val = val_data[y_list]
exog_x_val = val_data.drop(columns=y_list)

# Test
endog_y_test = test_data[y_list]
exog_x_test = test_data.drop(columns = y_list)

# Model with p = 1
VAR_model = VAR(endog_y)
results = VAR_model.fit(1)

# Predictions and residuals
predictions = results.forecast(endog_y.values, steps = endog_y.shape[0])
train_residuals = results.resid
train_residuals = pd.DataFrame(train_residuals)
train_residuals.columns = [ticker + "_residuals" for ticker in tickers]
train_residuals.to_csv('../output/VAR_results/VAR_train_residuals.csv', index=False)






# This function is only needed if we want to call training residuals from the VAR model.

def get_training_residual(data,tickers , p):
    """Given data, this function returns a pandas dataframe on the TRAINING residuals
        from a VAR model of order p. More precisely, suppose that we have training data
        of length m, e.g. in the above m = 0.8*n. We number our data points from earliest time
        to latest as 0, 1, .... , m-1. Then the function returns the residuals on time
        p-1, p, ...., m-1 (so the residual has m-p data points)."""
    VAR_model = VAR(data)
    results = VAR_model.fit(p)
    residuals = results.resid
    # Rename columns as residuals
    residuals.columns = [ticker + "_residuals" for ticker in tickers]
    return residuals

# Print training residuals for p = 2
#print(get_training_residual(endog_y, tickers, 2))



