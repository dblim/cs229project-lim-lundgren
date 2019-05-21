import pandas as pd

'''stock names:
['AMD','TRIP','AAP', 'RHT', 'CMG', 'ORLY', 'FOX', 'FOXA', 'HCA', 'BSX', 'MKC', 'ILMN', 'NRG', 'MRK', 'LLY', 'NFLX', 'EW', 'CRM','AES', 'RMD', 'CHD']
'''

def read_stock(name):
    path = 'top_stocks/'+name+'.csv'
    df = pd.read_csv(path)
    df = df.set_index('timestamp')
    return df

#sample usage
TRIP = read_stock('TRIP')
print(TRIP.head())