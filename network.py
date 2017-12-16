import numpy as np
from layer import FullyConnectedLayer
from utils import softmax

class NeuralNet:
    def __init__(self, din, dout, dhidden):
        self.l = len(dhidden)
        self.din = din
        self.dout = dout
        self.dhidden = dhidden
        self.layer = []
        self.layer.append(FullyConnectedLayer(din, dhidden[0]))
        for i in range(self.l - 1):
            self.layer.append(FullyConnectedLayer(dhidden[i], dhidden[i + 1]))
        self.layer.append(FullyConnectedLayer(dhidden[-1], dout))
        
        
        self.x  = [l.x  for l in self.layer]
        self.w  = [l.w  for l in self.layer]
        self.y  = [l.y  for l in self.layer]

        self.dx = [l.dx for l in self.layer]
        self.dw = [l.dw for l in self.layer]
        self.dy = [l.dy for l in self.layer]

    def __repr__(self):
        return 'NeuralNet with {0.din} to {0.dhidden} to {0.dout}'.format(self)

    def forward(self, x):
        for i in range(len(self.layer)):
            self.x[i] = x if i == 0 else np.maximum(0, self.y[i - 1])
            self.y[i] = self.layer[i].forward(self.x[i], self.w[i]) 
            #print(i)
            #print(self.x, self.w, self.y)
        

        return self.y[-1]

    def backward(self, dy):
        for i in reversed(range(len(self.layer))):
            self.dy[i] = dy if i == len(self.layer) - 1 else self.dx[i + 1] * (1 * self.y[i] > 0)
            self.dx[i], self.dw[i] = self.layer[i].backward(self.x[i], self.w[i], self.dy[i])
        return self.dw

    def loss(self, yhat, y):
        p = softmax(yhat).reshape(1, -1)
        loss = -(-10 if p[0][y] < np.exp(-10) else np.log(p[0][y]))
        dy = p
        dy[0][y] -= 1  
        return loss, dy

    def trainOnce(self, x, y, r):
        dwTotal = self.dw
        lossTotal = 0
        n = x.shape[0]
        for i in range(n):
            l, dy = self.loss(self.forward(x[i]), y[i])
            dw = self.backward(dy)
            
            lossTotal += l
            for j in range(len(dwTotal)):
                dwTotal[j] += dw[j]
        
        for j in range(len(dwTotal)):
            self.w[j] -= dwTotal[j] / n

        return lossTotal / n

    def train(self, x, y, iter, r):
        for t in range(iter):
            l = self.trainOnce(x, y, r)
            for j in range(len(self.layer)):
                self.layer[j].w = self.w[j]
            print(t, l)
    
    def test(self, x):
        y = []
        for i in range(x.shape[0]):
            yhat = self.forward(x[i])
            y.append(np.argmax(yhat))
        return np.array(y)