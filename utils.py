from PIL import Image

import numpy as np
import matplotlib.pyplot as plt 

def softmax(s):
    t = np.exp(s - s.max())
    return t / np.sum(t) 


def dataGen(n, d):
    x = np.random.normal(0, 1, [n, d])
    y = 1 * np.logical_or(np.logical_and(x[:, 0] > 0, x[:, 1] > 0), np.logical_and(x[:, 0] < 0, x[:, 1] < 0)) 
    for i in range(n):
        if np.sum(np.square(x[i])) > 2.0:
            y[i] = 2
        if np.sum(np.square(x[i])) < 1.0:
            y[i] = 3
    return x, y

def plot(x, y, category1, category2):
    plt.subplot(1, 2, 1)
    #c = np.array(['r', 'b', 'g', 'k'])
    #plt.plot(x, y, color=c[category1])

    plt.plot(x[category1==0], y[category1==0], color='b', marker='o',  linestyle='', markersize=1)
    plt.plot(x[category1==1], y[category1==1], color='r', marker='o',  linestyle='', markersize=1)
    plt.plot(x[category1==2], y[category1==2], color='g', marker='o',  linestyle='', markersize=1)
    plt.plot(x[category1==3], y[category1==3], color='k', marker='o',  linestyle='', markersize=1)

    plt.subplot(1, 2, 2)
    #plt.plot(x, y, color=c[category2])
    plt.plot(x[category2==1], y[category2==1], color='r', marker='o',  linestyle='', markersize=1)
    plt.plot(x[category2==0], y[category2==0], color='b', marker='o',  linestyle='', markersize=1)
    plt.plot(x[category2==2], y[category2==2], color='g', marker='o',  linestyle='', markersize=1)
    plt.plot(x[category2==3], y[category2==3], color='k', marker='o',  linestyle='', markersize=1)
    
    plt.show()

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

def toImg(rawArray):
    #print(rawArray.shape)
    H, W = 32, 32
    data = np.zeros((32, 32, 3), np.uint8)
    t = 0
    for c in range(3):
        for i in range(32):
            for j in range(32):
                #print(rawArray[t], i, j,  c)

                data[i][j][c] = rawArray[t]
                t += 1

    
    return Image.fromarray(data, 'RGB')

    d = unpickle("./data/data_batch_1")
    
    for k in d.keys():
        d[k.decode()] = d.pop(k)
    
    for i in range(10000):
        img = toImg(d['data'][i])
        img.show()

        input('next') 