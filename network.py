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
        
        self.l    = len(self.layer)

        self.w    = [l.generateWeight() for l in self.layer]
        
    def __repr__(self):
        return 'NeuralNet with {0.din} to {0.dhidden} to {0.dout}'.format(self)

    def forward(self, x0, w):
        x = [None] * self.l
        x[0] = x0
        i = 0
        while i < self.l - 1:
            x[i + 1] = np.maximum(0, self.layer[i].forward(x[i], w[i]))
            i += 1
        p = self.layer[i].forward(x[i], w[i])
        return x, p

    def backward(self, x, w, dy):
        dw = [None] * self.l

        for i in reversed(range(self.l)):
            dx, dw[i] = self.layer[i].backward(x[i], w[i], dy)      
            dy  = dx * (1 * (x[i] > 0))
        return dw

    def loss(self, s, y):
        prob = softmax(s).reshape(1, -1)
        THRESHOLD = -10
        loss = -(THRESHOLD if prob[0][y] < np.exp(THRESHOLD) else np.log(prob[0][y]))
        prob[0][y] -= 1
        return loss, prob

    def train_iteration(self, x, y, r):
        lossSum = 0
        correctSum = 0
        dwSum = None
        N = x.shape[0]
        w = self.w    
        for i in range(N):
            xi = x[i]
            yi = y[i]
            _x, _s = self.forward(xi, w)
            l, dy = self.loss(_s, yi)
            
            lossSum += l
            if yi == np.argmax(_s):
                correctSum += 1
            
            dw = self.backward(_x, w, dy)
            if dwSum is None:
                dwSum = dw
            else:
                for j in range(self.l)
                    dwSum[j] += dw[j] 

        for i in range(self.l):
            self.w[i] -= (dwSum[i] / N) * r

        return lossSum / N, correctSum / N

    def train(self, x, y, iter, r):
        for t in range(iter):
            l, correct_rate = self.train_iteration(x, y, r)
            s = 'Iter: {0:4d} | Loss: {1:2.2f} | CorrectRate: {2:2.2f} | StepSize:{3:2.2f}\r'.format(t, l, correct_rate, r)
            print(s, end='')
            print('\r', end='')
        print('\n')
    
    def test(self, x):
        y = []
        for i in range(x.shape[0]):
            yhat = self.forward(x[i], self.w)
            y.append(np.argmax(yhat[-1]))
        return np.array(y)

