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
        print(self.dim)
        for i in range(len(self.dim) - 1):
            self.layer.append(FullyConnectedLayer(self.dim[i], self.dim[i + 1]))
        
        self.l  = len(self.layer)
        self.w  = [l.initWeight()  for l in self.layer]
        self.dw = [l.dw  for l in self.layer]
        
    def __repr__(self):
        return 'NeuralNet with {0.din} to {0.dhidden} to {0.dout}'.format(self)

    def forward(self, _x, wArray):
        xArray = []
        x = np.array(_x)
        xArray.append(np.array(x))
        for i in range(self.l):
            y = self.layer[i].forward(x, wArray[i])
            if i != self.l - 1:
                y = np.maximum(0, y)

            xArray.append(np.array(y))
            x = y
        return xArray

    def backward(self, xArray, wArray, _dy):
        dwArray = []
        dy = np.array(_dy)
        for i in reversed(range(self.l)):
            dx, dw = self.layer[i].backward(xArray[i], wArray[i], dy)            
            dwArray.insert(0, np.array(dw))
            if i != self.l - 1:
                dx *= dx * (1 * (xArray[i] > 0))

            dy = dx
        return dwArray

    def loss(self, s, qIdx):
        p = softmax(s).reshape(1, -1)
        l = 10 if p[0][qIdx] < np.exp(-10) else -np.log(p[0][qIdx])
        p[0][qIdx] -= 1
        return l, p

    def trainOnce(self, x, y, r):
        dwArrayTotal = self.dw
        loss = 0
        correct = 0
        n = x.shape[0]
        wArray = self.w

        for dw in dwArrayTotal:
            dw.fill(0)

        for i in range(n):
            xArray  = self.forward(x[i], wArray)
            l, dy   = self.loss(xArray[-1], y[i])
            loss += l
            if y[i] == np.argmax(xArray[-1]):
                correct += 1

            dwArray = self.backward(xArray, wArray, dy)
            
            
            for j in range(self.l):
                dwArrayTotal[j] += dwArray[j]
        
        for i in range(self.l):
            self.w[j] -= dwArrayTotal[j] / n * r

        return loss / n, correct / n

    def train(self, x, y, iter, r):
        for t in range(iter):
            l, correct_rate = self.trainOnce(x, y, r)
            print(t, l, correct_rate)
    
    def test(self, x):
        y = []
        for i in range(x.shape[0]):
            yhat = self.forward(x[i], self.w)
            y.append(np.argmax(yhat[-1]))
        return np.array(y)