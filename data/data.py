import pickle
import numpy as np
import matplotlib.pyplot as plt 

def toy_data_gen(n, d):
    x = np.random.normal(0, 1, [n, d])
    y = 1 * np.logical_or(np.logical_and(x[:, 0] > 0, x[:, 1] > 0), np.logical_and(x[:, 0] < 0, x[:, 1] < 0)) 
    for i in range(n):
        if np.sum(np.square(x[i])) > 2.0:
            y[i] = 2
        elif np.sum(np.square(x[i])) < 1.0:
            y[i] = 3
     
    return x, y

def plot(x, y):
    color = ['r', 'y', 'g', 'c', 'b', 'p']
    for i in range(6):
        index = y == i
        plt.plot(x[index, 0], 
                 x[index, 1], 
                 c=color[i], 
                 marker='o', 
                 markersize=2, 
                 linestyle='none')
    plt.show()

def unpickle(file):
    with open(file, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return d

def array2img(array, c, h, w):            
    return array.reshape([c, h, w]) 

def cifar():
    f= './data/data_batch_1'
    d = unpickle(f)
    return np.array(d[b'data']), np.array(d[b'labels'])

def mnist():
    f = './data/mnist.pkl'
    d = unpickle(f)
    train, test, validata = d[0], d[1], d[2]
    return np.array(train[0]), np.array(train[1])



