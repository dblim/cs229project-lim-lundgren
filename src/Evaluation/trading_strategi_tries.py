import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import zero_one_loss
import seaborn as sns

tickers = ['ACN', 'AMAT', 'CDNS', 'IBM', 'INTU']#['LRCX', 'NTAP', 'VRSN', 'WU', 'XLNX']#

pred_path = '../output/LSTM_results/test_results/partial_all_stocks_pred.csv'
real_path = '../output/LSTM_results/test_results/partial_all_stocks_real.csv'

pred = pd.read_csv(pred_path).values
real = pd.read_csv(real_path).values
test = []
for i, ticker in enumerate(tickers):
    predicted_values = pred[:, i]
    real_values = real[:, i]
    MSE = sum(( real_values)**2)/predicted_values.shape[0]

    # accuracy
    real_zero_one = real_values.copy()
    real_zero_one[real_zero_one > 0] = 1
    real_zero_one[real_zero_one < 0] = 0
    pred_zero_one = predicted_values.copy()
    pred_zero_one[pred_zero_one > 0] = 1
    pred_zero_one[pred_zero_one < 0] = 0
    accuracy = 1 - zero_one_loss(pred_zero_one, real_zero_one)

    pred_zero_one = predicted_values.copy()
    FPR_list = [0]
    TRP_list = [0]
    list_of_t = sorted(predicted_values)
    auc = 0
    for j in range(len(list_of_t)):
        t = list_of_t[j]
        pred_zero_one = predicted_values.copy()
        pred_zero_one[pred_zero_one > t] = 1
        pred_zero_one[pred_zero_one < t] = 0
        TP = np.sum(np.logical_and(pred_zero_one == 1, real_zero_one == 1))
        TN = np.sum(np.logical_and(pred_zero_one == 0, real_zero_one == 0))
        FP = np.sum(np.logical_and(pred_zero_one == 1, real_zero_one == 0))
        FN = np.sum(np.logical_and(pred_zero_one == 0, real_zero_one == 1))
        TPR = TP/(TP + FN)
        FPR = FP/(FP + TN)
        auc += (FPR - FPR_list[j]) * TPR
        FPR_list.append(FPR)
        TRP_list.append(TPR)
    #print('AUC:', 1 - auc)
    #sns.set(color_codes=True)
    #plt.scatter(FPR_list, TRP_list, s=2, label=ticker+', AUC: '+str(round(1 - auc, 3)))
    #plt.plot([0, 1], [0, 1], color='red')

"""
plt.xlabel('False Positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curves')
plt.legend()
plt.savefig('../output/ROC_five_first.png')
plt.show()
"""

"""
    # returns
    predicted_values = pred[:, i]
    real_values = real[:, i]
    investment_strategy = np.zeros(predicted_values.shape)
    investment_strategy[predicted_values > 0] = real_values[predicted_values > 0]
    investment_strategy[predicted_values < 0] = real_values[predicted_values < 0]
    total_ret = 1
    for j in range(investment_strategy.shape[0]):
        total_ret *= (1 + investment_strategy[j])



    print('=====')
"""

    #print(ticker, np.mean(investment_strategy)/np.std(investment_strategy))

real_list = [1]
pred_list = [1]
idx = 1
tim_ad_real = []
tim_ad_pred = []
for i in range(int(real.shape[0])):
    real_list.append((real[i, idx] + 1) * real_list[i - 1])
    pred_list.append((pred[i, idx] + 1) * real_list[i - 1])
    if i % 12 == 0:
        tim_ad_real.append(real_list[i])
        tim_ad_pred.append(pred_list[i])

plt.plot(tim_ad_real, label='pred')
plt.plot(tim_ad_pred, label='real')

plt.legend()
plt.show()
