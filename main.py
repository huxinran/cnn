import numpy as np
import matplotlib.pyplot as plt 
from layer import FullyConnectedLayer 
from network import NeuralNet
from utils import dataGen
from utils import plot
from utils import softmax
import time


def main():
    print(1)

    N = 300
    din = 2
    dout = 4
    dh1 = 128
    dh2 = 128
    dhidden = [dh1, dh2]
    xArray, yArray = dataGen(N, din)
    r = 0.05
    T = 100



    np.random.seed(42)

    start = time.time()

    n = NeuralNet(din, dout, dhidden)    
    n.train(xArray, yArray, T, r)

    end = time.time()
    print(1)
    print(end - start)
    #return
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



        fc1.w -= dw1_total / N * r 
        fc2.w -= dw2_total / N * r
        fc3.w -= dw3_total / N * r
        
        #print('{0} {1} {2}               '.format(t, loss, correct),  end='')
        print('\r', end='')    
    
    #plot(xArray[:,0], xArray[:,1], yArray, c)
    end = time.time()
    print(1)
    print(end - start)
    return

if __name__ == "__main__":
    main()