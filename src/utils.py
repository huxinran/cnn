import pickle
import numpy as np
import matplotlib.pyplot as plt 

def softmax(y):
    p = np.exp(y - np.amax(y, axis=1))
    p /= np.sum(p, axis=1)
    return p

def softmax_cross_entropy_loss(y, lable):
    """
    given the output and true label,
    return softmax cross entropy loss and gradient on output
    """
    CAP = -10
    p = softmax(y)
    r = np.arange(label.size)
    loss = -np.log(np.maximum(np.exp(CAP), p[r, l]))
    dy = p
    dy[r, l] -= 1
    return loss, dy

def split(data, label, pct=0.0):
    """
    split data, label into train and test set
    """
    N = np.ceil(label.size * (1 - pct), dtype=int)
    data1, label1 = data[:N, :], label[:N, :]
    data2, label2 = data[N:, :], label[N:, :] 
    return data1, label1, data2, label2

def normalize(data):
    """
    normalize data
    """
    data = data - np.mean(data, axis=0)
    return data



