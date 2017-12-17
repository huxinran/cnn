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

def plot(x, y, label1, label2):
    color = np.array(['r', 'b', 'g', 'k'])
    
    plt.subplot(1, 2, 1)
    for i in range(4):
        plt.plot(x[label1==i], y[label1==i], c=color[i], marker='o', markersize=2, linestyle='none')

    plt.subplot(1, 2, 2)
    for i in range(4):
        plt.plot(x[label2==i], y[label2==i], c=color[i], marker='o', markersize=2, linestyle='none')

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


    np.random.seed(42)
    start = time.time()
    fc1 = FullyConnectedLayer(din, dh1)
    fc2 = FullyConnectedLayer(dh1, dh2)
    fc3 = FullyConnectedLayer(dh2, dout)

    print('\n')
    print('88888888888888888888888888888888')
    
    

    #plot(xArray[:,0], xArray[:,1], yArray)
    for t in range(T):
        dw1_total = np.zeros([din + 1, dh1])
        dw2_total = np.zeros([dh1 + 1, dh2])
        dw3_total = np.zeros([dh2 + 1, dout])

        w1 = fc1.generateWeight()
        w2 = fc2.generateWeight()
        w3 = fc3.generateWeight()    
        loss = 0.0
        correct = 0
        c = np.zeros(N)

        for i in range(N):
            x = xArray[i]
            y = yArray[i]

            x1 = x
            y1 = fc1.forward(x1, w1)
            x2 = np.maximum(0, y1)
            y2 = fc2.forward(x2, w2)

            x3 = np.maximum(0, y2)
            y3 = fc3.forward(x3, w3)
            p = softmax(y3)

            l = 10 if p[0][y] < 0.000001 else -np.log(p[0][y])
            
            c[i] = np.argmax(p)
            if c[i] == y:
                correct += 1
            
            loss += l

            dy3 = p
            dy3[0][y] -= 1
            dx3, dw3 = fc3.backward(x3, w3, dy3)

            dy2 = dx3 * (1 * y2 > 0)
            dx2, dw2 = fc2.backward(x2, w2, dy2)
            dy1 = dx2 * (1 * y1 > 0)
            dx1, dw1 = fc1.backward(x1, w1, dy1)


            dw1_total += dw1 
            dw2_total += dw2 
            dw3_total += dw3


        loss /= N
        correct /= N



        w1 -= dw1_total / N * r 
        w2 -= dw2_total / N * r
        w3 -= dw3_total / N * r
        
        #print('{0} {1} {2}               '.format(t, loss, correct),  end='')
        print('\r', end='')    
    
    #plot(xArray[:,0], xArray[:,1], yArray, c)
    end = time.time()
    print(1)
    print(end - start)
    return
