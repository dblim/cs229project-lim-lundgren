from utils import preprocess, lookback_kernel, quadratic_kernel, y_numeric_to_vector
import pandas as pd
import seaborn as sb
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

# Data source
path = 'data/top_stocks/AAP.csv'
data = pd.read_csv(path)
print(data['close'])

data['close'].plot()




