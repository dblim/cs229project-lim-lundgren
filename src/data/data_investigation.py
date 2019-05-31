import pandas as pd
import numpy as np
from utils import minutizer, combine_ts
import seaborn as sns
import matplotlib.pyplot as plt

tickers = ['AMD', 'TRIP', 'AAP', 'RHT', 'CMG', 'ORLY', 'FOX', 'FOXA', 'HCA', 'BSX', 'MKC', 'ILMN', 'NRG', 'MRK',
           'LLY', 'NFLX', 'EW', 'CRM', 'AES', 'RMD', 'CHD']


def return_vector(data):
    data = np.array(data)
    return data[1: data.shape[0]]/data[0: data.shape[0] - 1] - 1


def return_matrix(data, tickers):
    n, d = data.shape
    new_d = int(d/5)
    df = np.zeros((n - 1, new_d))
    for i in range(new_d):
        df[:, i] = return_vector(data.iloc[:, i*5])
    df = pd.DataFrame(df, columns=tickers)
    return df


data = minutizer(combine_ts(tickers))
return_matrix_a = return_matrix(data, tickers)

d = return_matrix_a.corr()
print(d)
mask = np.zeros_like(d, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(d, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.savefig('../data/correlation.png')

