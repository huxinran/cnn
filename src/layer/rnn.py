from layer import Layer
import utils
import numpy as np

class RNNLayer(Layer):
    def __init__(self, config):
        super(RNNLayer, self).__init__()
        self.type = 'RNN'
        self.config = config
        self.dim_hidden = config['dim_hidden']
        self.l = config['l']
        self.clip = config['clip']
        
    def accept(self, shape_in):      
        self.shape_in = shape_in 
        self.shape = self.shape_in
        self.dim_in = np.prod(self.shape_in, dtype=int)
        self.dim_out = np.prod(self.shape, dtype=int)
        # param 
        self.s0 = np.zeros([1, self.dim_hidden])
        self.U = np.random.normal(0, 1 / np.sqrt(self.dim_in), [self.dim_in, self.dim_hidden])
        self.W = np.random.normal(0, 1 / np.sqrt(self.dim_hidden), [self.dim_hidden, self.dim_hidden])
        self.V = np.random.normal(0, 1 / np.sqrt(self.dim_hidden), [self.dim_hidden, self.dim_out])
        self.bs = np.random.normal(0, 1 / np.sqrt(self.dim_hidden), [1, self.dim_hidden]) 
        self.by = np.random.normal(0, 1 / np.sqrt(self.dim_out), [1, self.dim_out])

        # cache
        self.dV = np.zeros_like(self.V)
        self.dW = np.zeros_like(self.W)
        self.dU = np.zeros_like(self.U)


    def sample(self, c, char2idx, idx2char):
        y = [None] * (self.l + 1)
        x = np.zeros([1, self.dim_in])
        x[0][char2idx[c]] = 1.0
        st = self.s0
        y[0] = c
        for t in range(self.l):
            st = np.tanh(x @ self.U + st @ self.W)
            x = st @ self.V
            idx = np.argmax(x)
            y[t + 1] = idx2char[idx]
        
        return ''.join(y)


    
    def forward(self, x):
        self.x = [None] * self.l
        self.h = [None] * self.l
        self.s = [None] * self.l
        y = [None] * self.l
        st = self.s0 
        for t in range(self.l):
            xt = x[t]
            ht = xt @ self.U + st @ self.W
            st = np.tanh(ht)
            yt = st @ self.V 

            self.x[t] = xt
            self.h[t] = ht
            self.s[t] = st
            y[t] = yt
        return y

    def backward(self, dy):
        dV = np.zeros_like(self.V)
        dW = np.zeros_like(self.W)
        dU = np.zeros_like(self.U)
        for t in reversed(range(self.l)):
            # for one time point only
            dVt = np.zeros_like(self.V)
            dWt = np.zeros_like(self.W)
            dUt = np.zeros_like(self.U)

            dyt = dy[t]
            dst, dVt, dbyt = utils.backward(dyt, self.s[t], self.V)
            dht = dst * (1 - self.h[t] ** 2)
            for i in reversed(range(t)):
                dxi, dUi, _ = utils.backward(dht, self.x[i], self.U)
                dsi, dWi, _ = utils.backward(dht, self.s[i], self.W)
                if t > 0:
                    dht = np.clip((dht @ self.W) * (1 - self.h[t - 1] ** 2), -self.clip, self.clip)
                else:
                    dht = np.clip((dht @ self.W), -self.clip, self.clip)

            dV += dVt
            dW += dWt
            dU += dUt
        
        return (dV, dW, dU)

    def learn(self, gradient):
        dVt, dWt, dUt = gradient
        self.dV = utils.compute_momentum(self.dV, dVt, self.config)
        self.dW = utils.compute_momentum(self.dW, dWt, self.config)
        self.dU = utils.compute_momentum(self.dU, dUt, self.config)
        self.V += self.dV
        self.W += self.dW
        self.U += self.dU
        

        #self.V -= self.config['step_size'] * dVt
        #self.W -= self.config['step_size'] * dWt
        #self.U -= self.config['step_size'] * dUt

    
    def fit(self, x, y, iter):
        l = len(x)
        i = 0
        while i < l:
            xb = x[i:i + self.l]
            yb = y[i:i + self.l]

            for t in range(iter):
                yhat = self.forward(xb)
                loss, dy = compute_rnn_loss(yhat, yb)
                grad = self.backward(dy)
                self.learn(grad)
                print(i, t, loss)
            
            self.s0 = self.s[-1]
            i += self.l

            
def compute_rnn_loss(yhat, y):
    l = len(y)
    loss = 0
    dy = [None] * l
    for t in range(l):
        pt = utils.softmax(yhat[t])
        losst, dy[t] = utils.cross_entropy(pt, y[t])
        loss += np.sum(losst)

    return loss, dy



