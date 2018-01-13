from layer import Layer
import numpy as np
import utils


class RNNLayer(Layer):
    def __init__(self, config):
        super(RNNLayer, self).__init__()
        self.type = 'RNN'
        self.config = config
        self.dim_hidden = config['dim_hidden']
        self.l = config['len']
        
    def accept(self, shape_in):      
        self.shape_in = shape_in
        self.shape = self.shape_in
        self.dim_in = np.prod(self.shape_in, dtype=int)
        self.dim_out = np.prod(self.shape, dtype=int)

        # param 
        self.W = np.random.normal(0, 1 / np.sqrt(self.dim_in), [self.dim_in, self.dim_hidden])
        self.U = np.random.normal(0, 1 / np.sqrt(self.dim_hidden), [self.dim_hidden, self.dim_hidden])
        self.V = np.random.normal(0, 1 / np.sqrt(self.dim_hidden), [self.dim_hidden, self.dim_out])
        self.bs = np.random.normal(0, 1, [1, self.dim_hidden]) 
        self.by = np.random.normal(0, 1, [1, self.dim_out])
        self.h0 = np.zeros([1, self.dim_hidden])

        #cache 
        self.x = [None] * self.l
        self.h = [None] * self.l
        self.s = [None] * self.l

    def sample(self, c, c2i, i2c):
        i = c2i[c]
        x = np.zeros([1, self.dim_in])
        x[0][i] = 1.0
        ht = self.h0
        y = [None] * (self.l + 1)
        y[0] = c
        for t in range(self.l):
            st = ht @ self.U + x @ self.W
            ht = np.tanh(st)
            yt = ht @ self.V
            yc = np.argmax(yt)
            y[t + 1] = i2c[yc]
            x = np.zeros([1, self.dim_in])
            x[0][yc] = 1.0        
        return ''.join(y)

    def forward(self, x):
        y = [None] * self.l
        ht = self.h0
        for t in range(self.l):
            xt = x[t]
            #print(ht.shape)
            #print(self.U.shape)
            #print(xt.shape)
            #print(self.W.shape)
            st = ht @ self.U + xt @ self.W + self.bs
            ht = np.tanh(st)
            yt = ht @ self.V + self.by

            self.x[t] = xt
            self.s[t] = st
            self.h[t] = ht
            y[t] = yt    
           # print(y[t])
        return y

    def backward(self, dy):
        dW = np.zeros_like(self.W)
        dU = np.zeros_like(self.U)
        dV = np.zeros_like(self.V)
        dby = np.zeros_like(self.by)
        dbs = np.zeros_like(self.bs)

        for t in reversed(range(self.l)):
            dyt = dy[t]
            dht, dVt, dbyt = utils.backward(dyt, self.h[t], self.V)
            dst = dht * (1 - self.s[t] * self.s[t])
            
            dWt = np.zeros_like(self.W)
            dUt = np.zeros_like(self.U)
            dbst = np.zeros_like(self.bs)
            for i in reversed(range(t)):
                dUt += self.h[i] * dst
                dWt += self.x[i].reshape(-1, 1) * dst
                dbst += dst
                dst = (self.U.T @ dst.T).T * (1 - self.s[i - 1] * self.s[i - 1])
                dst = np.clip(dst, -5, 5)

            dV += dVt
            dW += dWt
            dU += dUt
            dby += dbyt
            dbs += dbst
        
        dV = np.clip(dV, -5, 5)
        dW = np.clip(dW, -5, 5)
        dU = np.clip(dU, -5, 5)
        dby = np.clip(dby, -5, 5)
        dbs = np.clip(dbs, -5, 5)

        return dW, dU, dV, dby, dbs

    def compute_loss(self, yhat, y):
        loss = 0
        
        dy = [None] * self.l
        for t in range(self.l):
            losst, dy[t] = utils.cross_entropy(utils.softmax(yhat[t]), y[t])
            loss += np.sum(losst)

        return loss, dy

    def train_iteration(self, x, y):
        yhat = self.forward(x)
        loss, dy = self.compute_loss(yhat, y)
        dW, dU, dV, dby, dbs = self.backward(dy)
        
        return dW, dU, dV, dby, dbs, loss
    
    def fit(self, x, y, iter, config):
        dW = np.zeros_like(self.W)
        dU = np.zeros_like(self.U)
        dV = np.zeros_like(self.V)
        dby = np.zeros_like(self.by)
        dbs = np.zeros_like(self.bs)
        for t in range(iter):
            dWt, dUt, dVt, dbyt, dbst, loss = self.train_iteration(x, y)
            print(t, loss)
            dW = dW * 0.9 - dWt * config['step_size']
            dU = dU * 0.9 - dUt * config['step_size']
            dV = dV * 0.9 - dVt * config['step_size']
            dby = dby * 0.9 - dbyt * config['step_size']
            dbs = dbs * 0.9 - dbst * config['step_size']

            self.W += dW
            self.U += dU
            self.V += dV
            self.bs += dbs
            self.by += dby
