from statsmodels.tsa.statespace.varmax import VARMAX
from utils import minutizer, preprocess_2
import pandas as pd
from pandas import read_csv
from numpy import subtract

""" THIS IS WRONG!!!!"""


def varmax(ticker: str,
           p: int = 2,
           q: int = 0):
    data = preprocess_2(minutizer(read_csv('../data/Health-Care/' + ticker + '.csv', index_col='timestamp',
                                           parse_dates=True), split=5), ticker)
    n, d = data.shape

    # Split data
    train_val_test_split = {'train': 0.7, 'val': 0.85, 'test': 1}
    train_data = data[0: int(n*train_val_test_split['train'])]
    val_data = data[int(n*train_val_test_split['train']): int(n*train_val_test_split['val'])]
    test_data = data[int(n*train_val_test_split['val']): int(n*train_val_test_split['test'])]

    # split data in X and Y
    y_list = [ticker+'_returns']

    # Train
    endog_y = train_data[y_list]
    exog_x = train_data.drop(columns=y_list)

    # Validate
    endog_y_val = val_data[y_list]
    exog_x_val = val_data.drop(columns=y_list)

    # Test -- this is just here fore simplicity!
    endog_y_test = test_data[y_list]
    exog_x_test = test_data.drop(columns=y_list)

    # Fit model
    model = VARMAX(endog=endog_y.values, exog=exog_x.values)
    model_fit = model.fit(disp=False, order=(p, q))

    # validate
    predictions_val = model_fit.forecast(steps=exog_x_val.shape[0], exog=exog_x_val.values)
    print('MSE:', sum(sum(subtract(predictions_val, endog_y_val)**2)))

    # Test -- NOTE that this is just here for simplicity
    predictions_test = model_fit.forecast(steps=exog_x_test.shape[0], exog=exog_x_test.values)

    # Save
    pd.DataFrame(predictions_val).to_csv('../output/ARIMA_results/predictions/val_files/'+ticker+'_val_predictions.csv',
                                         index=False)
    pd.DataFrame(endog_y_val).to_csv('../output/ARIMA_results/predictions/val_files/'+ticker+'_val_real.csv',
                                     index=False)
    pd.DataFrame(predictions_test).to_csv('../output/ARIMA_results/predictions/test_files/'+ticker+'_test_predictions.csv',
                                          index=False)
    pd.DataFrame(endog_y_test).to_csv('../output/ARIMA_results/predictions/test_files/'+ticker+'_test_real.csv',
                                      index=False)