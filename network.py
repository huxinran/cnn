import numpy as np
from layer import FullyConnectedLayer as FC
from utils import softmax
from utils import plot
import time

class NeuralNet:
    '''
    Neural Net class consists of l fully connected layers that classify a din dimensional data into one of 
    dout class labels

    dim  : dimension of data flow  [din, dh1, dh2, ..., dhk, dout]
    '''
    def __init__(self, dim):    
        self.dim = dim
        self.l = len(self.dim) - 1
        self.w = [None] * self.l
        for i in range(self.l):
            self.w[i] = FC.initWeight(self.dim[i], self.dim[i + 1])
        
        
    def __repr__(self):
        return 'NeuralNet with {0.din} to {0.dhidden} to {0.dout}'.format(self)

    def predict(self, xInput):
        xHidden = [None] * self.l
        xHidden[0] = xInput
        
        for i in range(self.l - 1):
            xHidden[i + 1] = np.maximum(0, FC.predict(xHidden[i], self.w[i]))
        
        y = FC.predict(xHidden[-1], self.w[-1])
        return y, xHidden

    def gradient(self, dy, xHidden):
        dwHidden = [None] * self.l

        for i in reversed(range(self.l)):
            dy, dwHidden[i] = FC.gradient(dy, xHidden[i], self.w[i])      
            dy = dy * (1 * (xHidden[i] > 0))

        for dw in dwHidden:
            dw /= dy.shape[0]

        return dwHidden

    def applyGraident(self, gradient, stepSize=0.001, regularization=0):
        for i in range(self.l):
            self.w[i] -= (gradient[i] * stepSize + self.w[i] * regularization)

    def calcLoss(self, y, label):
        p = softmax(y)
        r = np.arange(y.shape[0])
        loss = -np.log(np.maximum(np.exp(-10), p[r, label]))
        p[r, label] -= 1
        return loss, p

    def trainIteration(self, data, label, debug=False):
        # forawrd feed, get y and x_hidden
        y, xHidden = self.predict(data)
        
        # measure loss and gradient on y
        loss, dy = self.calcLoss(y, label)
        
        # backprop, get gradient on weight
        dw = self.gradient(dy, xHidden)
        
        if debug:
            print("data=", data)
            print("log=", label)
            print('w=', self.w)
            print("xhidden=", xHidden)
            print("y=", y)
            print("loss=", loss)
            print("dy=", dy)
            print("dw=", dw)

        return y, loss, dw

    def train(self, data, label, iteration, stepSize=0.001, regularization=0.0, testPct=0.0, debug=False):
        start = time.time()
        
        for t in range(iteration):
            y, loss, dw = self.trainIteration(data, label, debug)

            self.applyGraident(dw, stepSize, regularization)

            avgLoss = np.mean(loss)

            errRate = np.mean(1 * (np.argmax(y, axis=1) != label))

            timeRemain = (time.time() - start) / (t + 1) * (iteration - t - 1)
            
            debugStr = 'Iter: {0:4d} | Loss: {1:4.4f} | Train ErrRate: {2:4.4f} | Time Remain:{3:4.4f}'.format(t, avgLoss, errRate, timeRemain)
            print(debugStr, end='')
            print('\r', end='')
            
        print('\nTime total : {0}'.format(time.time() - start))


    def show(self):
        x = np.linspace(-4, 4, 128)
        y = np.linspace(-4, 4, 128)
        data = []
        for xi in x:
            for yi in y:
                data.append([xi, yi])
        
        data = np.array(data)
        y, _ = self.predict(data)
        plot(data[:,0], data[:,1], np.argmax(y, axis=1))