from layer import Layer
import utils
import numpy as np

class RNNLayer(Layer):
    def __init__(self, config):
        super(RNNLayer, self).__init__()
        self.type = 'RNN'
        self.config = config
        self.dim_hidden = config['dim_hidden']
        self.clip = config['clip']
        
    def accept(self, shape_in):      
        self.shape_in = shape_in 
        self.shape = self.shape_in
        self.dim_in = np.prod(self.shape_in, dtype=int)
        self.dim_out = np.prod(self.shape, dtype=int)
        # param 
        self.h0 = np.zeros([1, self.dim_hidden])
        
        self.U = np.random.randn(self.dim_in, self.dim_hidden) / np.sqrt(self.dim_in)
        self.W = np.random.randn(self.dim_hidden, self.dim_hidden) / np.sqrt(self.dim_hidden)
        self.bh = np.zeros([1, self.dim_hidden]) 

        self.V = np.random.randn(self.dim_hidden, self.dim_out) / np.sqrt(self.dim_hidden)
        self.by = np.zeros([1, self.dim_out])
        
        # cache
        self.dV = np.zeros_like(self.V)
        self.dW = np.zeros_like(self.W)
        self.dU = np.zeros_like(self.U)
        self.dby = np.zeros_like(self.by)
        self.dbh = np.zeros_like(self.bh)
    
    def forward(self, x, h_prev):
        l = len(x)
        self.x = [None] * l
        self.s = [None] * l
        self.h = [None] * l
        y = [None] * l
        ht = h_prev
        for t in range(l):
            xt = x[t]
            st = xt @ self.U + ht @ self.W + self.bh
            ht = np.tanh(st)
            yt = ht @ self.V + self.by

            self.x[t] = xt
            self.s[t] = st
            self.h[t] = ht
            y[t] = yt
        return y

    def backward(self, dy, dh):
        l = len(dy)
        dV = np.zeros_like(self.V)
        dW = np.zeros_like(self.W)
        dU = np.zeros_like(self.U)
        dby = np.zeros_like(self.by)
        dbh = np.zeros_like(self.bh)
        
        dh_prev = dh
        for t in reversed(range(l)):
            # for one time point only
            dVt = np.zeros_like(self.V)
            dWt = np.zeros_like(self.W)
            dUt = np.zeros_like(self.U)

            dyt = dy[t]
            dht, dVt, dbyt = utils.backward(dyt, self.h[t], self.V)
            dht = dht + dh_prev 
            
            dst = dht * (1 - self.h[t] ** 2)
            dbht = dst
            #print(t)

            if t == 0:
                dh_prev, dWt, _ = utils.backward(dst, self.h[t - 1], self.W)
            else:
                dh_prev, dWt, _ = utils.backward(dst, self.h0, self.W)

            dxt, dUt, _ = utils.backward(dst, self.x[t], self.U)

            dV += dVt
            dW += dWt
            dU += dUt
            dby += dbyt
            dbh += dbht

        
        dV = np.clip(dV, -self.clip, self.clip)
        dW = np.clip(dW, -self.clip, self.clip)
        dU = np.clip(dU, -self.clip, self.clip)
        dby = np.clip(dby, -self.clip, self.clip)
        dbh = np.clip(dbh, -self.clip, self.clip)

        return dV, dW, dU, dby, dbh, dh_prev

    def learn(self, gradient):
        dVt, dWt, dUt, dbyt, dbht, dh_prev = gradient
        self.dV += dVt ** 2
        self.dW += dWt ** 2
        self.dU += dUt ** 2 
        self.dby += dbyt ** 2
        self.dbh += dbht ** 2 

        step_size = self.config['step_size']
        self.V -= dVt * self.dV ** -0.5 * step_size
        self.W -= dWt * np.sqrt(self.dW) * step_size
        self.U -= dUt * np.sqrt(self.dU) * step_size
        self.by -= dbyt * np.sqrt(self.dby) * step_size
        self.bh -= dbht * np.sqrt(self.dbh) * step_size

        
    def sample(self, c, l, char2idx, idx2char):
        y = [None] * (l + 1)
        x = np.zeros([1, self.dim_in])
        x[0][char2idx[c]] = 1.0
        ht = self.h0
        y[0] = c
        for t in range(l):
            ht = np.tanh(x @ self.U + ht @ self.W + self.bh)
            yhat = ht @ self.V + self.by
            idx = np.argmax(yhat)
            y[t + 1] = idx2char[idx]
        
        return ''.join(y)
        #self.V -= self.config['step_size'] * dVt
        #self.W -= self.config['step_size'] * dWt
        #self.U -= self.config['step_size'] * dUt

    def translate(self, yhat, idx2char):
        return ''.join([idx2char[np.argmax(y)] for y in yhat])

    
    def fit(self, x, y, l, iter, char2idx, idx2char):
        lx = len(x)
        for t in range(iter):
            i = 0
            ht = self.h0
            dh_prev = self.h0
            ##if t % 100 == 0:
             #   self.config['step_size'] *= 0.8

            while i < lx:
                e = min(lx, i + l)
                xb = x[i:e]
                yb = y[i:e]
                yhat = self.forward(xb, ht)
                loss, dy = compute_rnn_loss(yhat, yb)
                grad = self.backward(dy, dh_prev)
                self.learn(grad)
                ht = self.h[-1]
                dh_prev = grad[-1]
                tt = self.translate(yhat, idx2char)
                print(t, loss, tt)
                i += l
            
            
def compute_rnn_loss(yhat, y):
    l = len(y)
    loss = 0
    dy = [None] * l
    for t in range(l):
        pt = utils.softmax(yhat[t])
        losst, dy[t] = utils.cross_entropy(pt, y[t])
        loss += np.sum(losst)

    return loss, dy



