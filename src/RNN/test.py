import pandas as pd
path = '../data/top_stocks/CRM.csv'
data = pd.read_csv(path, index_col="timestamp", parse_dates=True)

def data_divider(data, integer):
    pass
print(data.iloc[0:2])