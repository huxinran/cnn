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
        self.model = {
            'U' : np.random.randn(self.dim_in, self.dim_hidden) * 0.01
          , 'W' : np.random.randn(self.dim_hidden, self.dim_hidden) * 0.01
          , 'V' : np.random.randn(self.dim_hidden, self.dim_out) * 0.01 
          , 'bh' : np.zeros([1, self.dim_hidden])
          , 'by' : np.zeros([1, self.dim_out])
        }

        self.G = {
            'U' : np.zeros([self.dim_in, self.dim_hidden])
          , 'W' : np.zeros([self.dim_hidden, self.dim_hidden])
          , 'V' : np.zeros([self.dim_hidden, self.dim_out]) 
          , 'bh' : np.zeros([1, self.dim_hidden])
          , 'by' : np.zeros([1, self.dim_out])
        }
        # cache
        self.x = []
        self.h = []

    def forward(self, xs, h_prev, V, U, W, bh, by):
        l = len(xs)
        ys = []
        self.x = []
        self.h = []
        h = np.copy(h_prev)
        for t in range(l):
            s = xs[t] @ U + h @ W + bh
            h = np.tanh(s)
            y = h @ V + by
            self.x.append(xs[t])
            self.h.append(h)
            ys.append(y)
        return ys

    def backward(self, dys, h_prev_i, V, W, U, by, bh):
        grad = {k:np.zeros_like(v) for k, v, in self.model.items()}
        l = len(dys)
        dh_prev = np.zeros([1, self.dim_hidden])
        for t in reversed(range(l)):            
            if t == 0:
                h_prev = np.copy(h_prev_i)
            else:
                h_prev = self.h[t - 1]

            dy = np.copy(dys[t])
            dh, dV, dby = utils.backward(dy, self.h[t], V)
            dh += dh_prev
            ds = dh * (1 - self.h[t] ** 2)
            dbh = ds
            dx, dU, _ = utils.backward(ds, self.x[t], U) 
            dh_prev, dW, _ =utils.backward(ds, h_prev, W)
            grad['V'] += dV
            grad['U'] += dU
            grad['W'] += dW
            grad['bh'] += dbh
            grad['by'] += dby
        
        for k, v in grad.items():
            v = np.clip(v, -self.clip, self.clip)

        return grad, dh_prev

    def learn(self, grad):
        for k, v in grad.items():
            self.G[k] += v ** 2

        step_size = self.config['step_size']
        for k, v in grad.items():
            self.model[k] -= v / np.sqrt(np.maximum(1e-10, self.G[k]))

        
    def sample(self, c, l, char2idx, idx2char, V, U, W, bh, by):
        ll = len(char2idx)
        yp = [np.zeros([1, ll])] * l
        x = np.zeros([1, self.dim_in])
        x[0][char2idx[c]] = 1.0

        h = np.zeros([1, self.dim_hidden])
        for i in range(100):
            for t in range(l):
                h = np.tanh(x @ U + h @ W + bh)
                yhat = h @ V + by
                ypt = utils.softmax(yhat)
                idx = np.random.choice(ll, 1, p=ypt[0])
                
                yp[t] += ypt
                x = np.zeros([1, self.dim_in])
                x[0][idx] = 1.0
                c = [idx2char[np.argmax(p)] for p in ypt]
                print(''.join(c))
            
        
            

        return ''.join(c)
        #self.V -= self.config['step_size'] * dVt
        #self.W -= self.config['step_size'] * dWt
        #self.U -= self.config['step_size'] * dUt



    
    def fit(self, x, y, l, iter, char2idx, idx2char):
        lx = len(x)
        for t in range(iter):
            i = 0
            
            h = np.zeros([1, self.dim_hidden])
            loss = 0
            while i < lx:
                e = min(lx, i + l)
                xb = x[i:e]
                yb = y[i:e]
                yhat = self.forward(xb, h, **self.model)
            
                p = [utils.softmax(y) for y in yhat]
                losst, dy = utils.cross_entropy_list(p, yb)
                loss += losst
                
                grad, _ = self.backward(dy, h, **self.model)
                
                self.learn(grad)
                h = np.copy(self.h[-1])

                i += l
                
            if t % 100 == 0:
                tt = self.sample('f', 40, char2idx, idx2char, **self.model)
                print(t, loss, tt)            




