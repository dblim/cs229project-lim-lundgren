import numpy as np
A = [1,3,5,7]
temp = A

def first_difference(data):
    temp = np.zeros(len(data))
    temp[0] = data[0]
    for i in range(1,len(data)):
        temp[i] = data[i-1] - data[i]
    return temp

def difference(data,k=1):
    temp = data
    for i in range(k):
        temp = first_difference(temp)
    return temp

def moving_mean(data,start,stop):
    return np.mean(data[start:stop])

def moving_cov(data,start,stop):
    return np.var(data[start:stop])


#for i in range(4):
    #print(difference(A,i))


