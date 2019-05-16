import numpy as np
import random

"""
Given data, compute the mean phi

bernoulli variable = phi+imporvement

model: randomly pick success with probability phi+imporvement
"""

def bernoulli_model(p):
    #p is probability of success
    random_val = random.uniform(0,1)
    if random_val<p:
        return 1
    else:
        return 0

def fit(y,improvement):
    return np.mean(y)+improvement

#In the accuracy function, p should be fit(y, improvement) where improvement is some predetermined number.

def bernoulli_accuracy(y,p):
    #generate predictions for p
    predictions = np.array([bernoulli_model(p) for i in range(len(y))])
    return ('accurary?',np.mean(np.linalg.norm(y - predictions, ord = 1)),'predictions is', predictions)

