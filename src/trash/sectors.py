import csv
import pandas as pd
import seaborn as sns
from utils import minutizer, combine_ts_with_path
import numpy as np
import matplotlib.pyplot as plt

'''To run different sectors you just have to go to 
line 38 an input a differnt string for the sector
'''


# read in sector_dict.csv and convert into dictionary
def process_dict_values(s):
    # s is a string that needs to be converted into list
    tt = s[2:-2].replace(' ','').replace("'",'').replace('"','')
    return tt.split(',')

with open('data/sector_dict.csv') as csv_file:
    reader = csv.reader(csv_file)
    sector_dict ={row[0]: process_dict_values(row[1]) for row in reader}

'''
keys in sector_dict:
Industrials
Health Care
Information Technology
Communication Services
Consumer Discretionary
Utilities
Financials
Materials
Real Estate
Consumer Staples
Energy
'''

sector = 'Information Technology'
tickers = sector_dict[sector]

# test_run of code
# doesn't take as long to run as a real sector:
''' 
sector = 'zzz'
tickers = ['AAP','AAPL','AIV','ATVI']
'''

base_path = 'data/sectors/'+sector+'/'

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

data = minutizer(combine_ts_with_path(base_path,tickers))
print('finished with minuteizer...')

return_matrix_a = return_matrix(data, tickers)
d = return_matrix_a.corr()



mask = np.zeros_like(d, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(d, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# save figure
plt.savefig('data/corr_plots/'+sector+'_table'+'.png')

# save corr values to a file
np.savetxt('data/corr_plots/'+sector+'_corr.txt',d)