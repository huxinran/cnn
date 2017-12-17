import numpy as np
from layer import FullyConnectedLayer
from utils import softmax

class NeuralNet:
    def __init__(self, din, dout, dhidden):
        self.dim = []
        self.dim.append(din)
        self.dim.extend(dhidden)
        self.dim.append(dout)

        self.layer = []
        for i in range(len(self.dim) - 1):
            self.layer.append(FullyConnectedLayer(self.dim[i], self.dim[i + 1]))
        
        self.l  = len(self.layer)
        self.w  = [l.w for l in self.layer]
        
        
    def __repr__(self):
        return 'NeuralNet with {0.din} to {0.dhidden} to {0.dout}'.format(self)

    def forward(self, _x, wArray):
        xArray = [np.array(1)] * (self.l + 1)
        xArray[0] = _x
        for i in range(self.l):
            xArray[i + 1] = self.layer[i].forward(xArray[i], wArray[i])
            if i != self.l - 1:
                xArray[i + 1] = np.maximum(0, xArray[i + 1])
        return xArray

    def backward(self, xArray, wArray, _dy):
        dwArray = [None] * self.l
        dy = np.array(_dy)
        for i in reversed(range(self.l)):
            dx, dw = self.layer[i].backward(xArray[i], wArray[i], dy)            
            dy = dx * (1 * (xArray[i] > 0))
            dwArray[i] = dw
            

        return dwArray

    def loss(self, s, y):
        p = softmax(s).reshape(1, -1)
        l = 10 if p[0][y] < 0.000001 else -np.log(p[0][y])
        p[0][y] -= 1
        return l, p

    def trainOnce(self, x, y, r):
        dwArrayTotal = None
        loss = 0
        correct = 0
        n = x.shape[0]
        wArray = self.w

        for i in range(n):
            xArray  = self.forward(x[i], wArray)
            l, dy   = self.loss(xArray[-1], y[i])
            loss += l
            if y[i] == np.argmax(xArray[-1]):
                correct += 1
            dwArray = self.backward(xArray, wArray, dy)

            if dwArrayTotal is None:
                dwArrayTotal = dwArray 
            else:
                for j in range(self.l):
                    dwArrayTotal[j] += dwArray[j]
        
        for i in range(self.l):
            self.w[i] -= (dwArrayTotal[i] / n) * r

        return loss / n, correct / n

    def train(self, x, y, iter, r):
        for t in range(iter):
            l, correct_rate = self.trainOnce(x, y, r)
            #s = 'Iter: {0:4d} | Loss: {1:2.2f} | CorrectRate: {2:2.2f} | StepSize:{3:2.2f}\r'.format(t, l, correct_rate, r)
            #print('\r', end='')
            #print(s, end='')
    
    def test(self, x):
        y = []
        for i in range(x.shape[0]):
            yhat = self.forward(x[i], self.w)
            y.append(np.argmax(yhat[-1]))
        return np.array(y)