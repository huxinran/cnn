import numpy as np
import matplotlib.pyplot as plt 
from layer import FullyConnectedLayer 

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


def main():
    N = 300
    din = 2
    dout = 4
    dh1 = 30
    dh2 = 30

    fc1 = FullyConnectedLayer(din, dh1)
    fc2 = FullyConnectedLayer(dh1, dh2)
    fc3 = FullyConnectedLayer(dh2, dout)

    xArray, yArray = dataGen(N, din)
    T = 100


    #plot(xArray[:,0], xArray[:,1], yArray)
    for t in range(T):
        dw1_total = np.zeros([din + 1, dh1])
        dw2_total = np.zeros([dh1 + 1, dh2])
        dw3_total = np.zeros([dh2 + 1, dout])

        w1 = fc1.w
        w2 = fc2.w
        w3 = fc3.w    
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

            l = -np.where(p[y] < 0.000001, -10, np.log2(p[y]))
            
            c[i] = np.argmax(p)
            if c[i] == y:
                correct += 1
            
            loss += l

            dy3 = p
            dy3[y] -= 1
            dx3, dw3 = fc3.backward(x3, w3, dy3)

            dy2 = dx3 * (1 * y2 > 0)
            dx2, dw2 = fc2.backward(x2, w2, dy2)
            
            dy1 = dx2 * (1 * y1 > 0)
            dx1, dw1 = fc1.backward(x1, w1, dy1)
            
            dw1_total += dw1 
            dw2_total += dw2 
            dw3_total += dw3


        r = 0.01
        loss /= N
        correct /= N
        dw1_total /= N
        dw2_total /= N
        dw3_total /= N
        fc1.w -= dw1_total * r 
        fc2.w -= dw2_total * r
        fc3.w -= dw3_total * r
        print (t, loss, correct)    
    
    plot(xArray[:,0], xArray[:,1], yArray, c)
        
    return

if __name__ == "__main__":
    main()