import pandas as pd
from pandas import Series
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR

# Data source
path = 'data/top_stocks/AAP.csv'
data = pd.read_csv(path)
frame = Series.from_csv(path, header = 0)
X = frame.values
n = len(X)

# Train/val split:

train_data = X[ : int(0.8*n)]
val_data = X[int(0.8*n) : ]

"""# Fit model
exog = train_data
model = AR(train_data.astype(float))
model_fit = model.fit()

# Prediction
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)"""



