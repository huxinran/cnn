import pickle
import numpy as np
import matplotlib.pyplot as plt 

def softmax(y):
    p = np.exp(y - np.amax(y, axis=1).reshape(-1, 1))
    return p / np.sum(p, axis=1).reshape(-1, 1) 


def dataGen(n, d):
    x = np.random.normal(0, 1, [n, d])
    y = 1 * np.logical_or(np.logical_and(x[:, 0] > 0, x[:, 1] > 0), np.logical_and(x[:, 0] < 0, x[:, 1] < 0)) 
    for i in range(n):
        if np.sum(np.square(x[i])) > 2.0:
            y[i] = 2
        elif np.sum(np.square(x[i])) < 1.0:
            y[i] = 3
     
    return x, y

def plot(x, y, label1, label2 = None):
    color = np.array(['r', 'b', 'g', 'k', 'y', 'c'])
    
    plt.subplot(1, 2, 1)
    for i in range(6):
        plt.plot(x[label1==i], y[label1==i], c=color[i], marker='o', markersize=2, linestyle='none')

    if label2 is not None:
        plt.subplot(1, 2, 2)
        for i in range(6):
            plt.plot(x[label2==i], y[label2==i], c=color[i], marker='o', markersize=2, linestyle='none')

    plt.show()

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

def cifar():
    f= './data/data_batch_1'
    d = unpickle(f)
    return np.array(d[b'data']), np.array(d[b'labels'])

def mnist():
    f = './data/mnist.pkl'
    d = unpickle(f)
    train, test, validata = d[0], d[1], d[2]
    return train

def toImg(rawArray, h, w, c):
    #print(rawArray.shape)
    data = np.zeros((h, w, c), np.uint8)
    t = 0
    for c in range(c):
        for i in range(h):
            for j in range(w):
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



def debug():
    start = time()

    print('\n')
    print('88888888888888888888888888888888')
    np.random.seed(seed)
    w1 = FC.initWeight(din, dh1)
    w2 = FC.initWeight(dh1, dh2)
    w3 = FC.initWeight(dh2, dout) 

    #plot(xArray[:,0], xArray[:,1], yArray)
    for t in range(iteration):
        dw1_total = np.zeros([din + 1, dh1])
        dw2_total = np.zeros([dh1 + 1, dh2])
        dw3_total = np.zeros([dh2 + 1, dout])

        loss = 0.0
        correct = 0
        c = np.zeros(N)

        for i in range(N):
            x = data[i].reshape([1, -1])
            y = label[i]

            x1 = x
            y1 = FC.predict(x1, w1)
            
            x2 = np.maximum(0, y1)
            y2 = FC.predict(x2, w2)

            x3 = np.maximum(0, y2)
            y3 = FC.predict(x3, w3)
            p = softmax(y3)

            l = 10 if p[0][y] < 0.000001 else -np.log(p[0][y])
            


            c[i] = np.argmax(p)
            if c[i] == y:
                correct += 1
            
            loss += l

            dy3 = p
            dy3[0][y] -= 1
            dx3, dw3 = FC.gradient(dy3, x3, w3)

            dy2 = dx3 * (1 * y2 > 0)
            dx2, dw2 = FC.gradient(dy2, x2, w2)
            
            dy1 = dx2 * (1 * y1 > 0)
            dx1, dw1 = FC.gradient(dy1, x1, w1)



            dw1_total += dw1 
            dw2_total += dw2 
            dw3_total += dw3

            if debug:
                print("x1=",x1, "x2=",x2, "x3=",x3)
                print("w1=",w1, "w2=",w2, "w3=",w3)
                print("y=", y3, 'p=',p, 'l=',  l)
            
                print("dy1=",dy1, "dy2=",dy2, "dy3=",dy3)
                print("dw1=",dw1, "dw2=",dw2, "dw3=",dw3)
                print("dx1=",dx1, "dx2=",dx2, "dx3=",dx3)
            

        loss /= N
        correct /= N



        w1 -= dw1_total / N * stepSize
        w2 -= dw2_total / N * stepSize
        w3 -= dw3_total / N * stepSize
        
        print('{0} {1} {2}               '.format(t, loss, correct),  end='')
        print('\r', end='')    
    
    #plot(xArray[:,0], xArray[:,1], yArray, c)
    end = time()
    print(1)
    print(end - start)
    return
