from utils import combine_ts_returns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

"""In this script, we will get the training residuals for VAR"""

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

print(get_training_residual(endog_y, tickers, 2))




#print(results.fittedvalues[:2])
#print(endog_y.iloc[:2])
#print(residuals.iloc[:2])






# predictions = prediction_function(endog_y.values, 10)

# Prints MSE error for each p.
"""for p in range(len(predictions)):
    for i, ticker in  enumerate( tickers):
        MSE = sum((predictions[p-1][:, i] - endog_y_val.values[:, i])**2)/endog_y_val.shape[0]
        print("For p = {} and {}, ".format(p+1, ticker) + "MSE error is", MSE)"""

"""def prediction_function(values, p):
    model = VAR(values)
    return [ model.fit(i).forecast(values, steps=10) for i in range(1,p)]"""

