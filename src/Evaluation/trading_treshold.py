import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import zero_one_loss
import seaborn as sns

pred_path = '../output/LSTM_results/test_results/partial_all_stocks_pred.csv'
real_path = '../output/LSTM_results/test_results/partial_all_stocks_real.csv'

pred = pd.read_csv(pred_path).values
real = pd.read_csv(real_path).values
print(real.shape)
market_returns = np.mean(real, axis=1)

tuner = sorted(pred.flatten())
print(len(tuner))
tun_list = []
ret_list_short = []
ret_list = []

for i in range(len(tuner)-3):
    t = tuner[i]
    print(i)
    strategy_returns_short = np.zeros(real.shape)
    strategy_returns = np.zeros(real.shape)
    for i in range(strategy_returns_short.shape[0]):
        for j in range(strategy_returns_short.shape[1]):
            if pred[i, j] > t:
                strategy_returns_short[i, j] = real[i, j]
                strategy_returns[i, j] = real[i, j]
            else:
                strategy_returns_short[i, j] = -real[i, j]
    strategy_returns_short = np.mean(strategy_returns_short, axis=1)
    strategy_returns = np.mean(strategy_returns, axis=1)

    #R = 1
    #for j in range(strategy_returns.shape[0]):
    #    R *= (1 + strategy_returns[j])

    tun_list.append(t)
    ret_list_short.append(np.mean(strategy_returns_short)/np.std(strategy_returns_short))
    ret_list.append(np.mean(strategy_returns)/np.std(strategy_returns))

sns.set(color_codes=True)
plt.plot(tun_list, ret_list_short, label='Strategy: long/short')
plt.plot(tun_list, ret_list, label='Strategy: long/pass')
plt.title('Optimal threshold')
plt.xlabel('Threshold')
plt.ylabel('Sharpe ratio of portfolio')
plt.legend()
plt.savefig('../output/beta_treshold_on_test_set.png')
plt.show()