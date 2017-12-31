from layer import Layer
import utils
import numpy as np

class LSTMLayer(Layer):
    def __init__(self, config):
        super(LSTMLayer, self).__init__()
        self.type = 'LSTM'
        self.config = config
        self.dim_hidden = config['dim_hidden']
        self.clip = config['clip']
        
    def accept(self, shape_in):      
        self.shape_in = shape_in 
        self.shape = self.shape_in
        self.dim_in = np.prod(self.shape_in, dtype=int)
        self.dim_out = np.prod(self.shape, dtype=int)
        
        # param 
        self.c0 = np.zeros([1, self.dim_hidden])

        self.Uf = np.random.randn(self.dim_in, self.dim_hidden) / np.sqrt(self.dim_in)
        self.Wf = np.random.randn(self.dim_hidden, self.dim_hidden) / np.sqrt(self.dim_hidden)
        self.bf = np.zeros((1, self.dim_hidden))
        self.Ui = np.random.randn(self.dim_in, self.dim_hidden) / np.sqrt(self.dim_in)
        self.Wi = np.random.randn(self.dim_hidden, self.dim_hidden) / np.sqrt(self.dim_hidden)
        self.bi = np.zeros((1, self.dim_hidden))
        self.Uo = np.random.randn(self.dim_in, self.dim_hidden) / np.sqrt(self.dim_in)
        self.Wo = np.random.randn(self.dim_hidden, self.dim_hidden) / np.sqrt(self.dim_hidden)
        self.bo = np.zeros((1, self.dim_hidden))
        self.Uc = np.random.randn(self.dim_in, self.dim_hidden) / np.sqrt(self.dim_in)
        self.Wc = np.random.randn(self.dim_hidden, self.dim_hidden) / np.sqrt(self.dim_hidden)
        self.bc = np.zeros((1, self.dim_hidden))
        self.V = np.random.randn(self.dim_hidden, self.dim_out) / np.sqrt(self.dim_hidden)
        self.by = np.zeros((1, self.dim_out))

        # cache
    
    def forward(self, x):
        l = len(x)
        self.c = [None] * l # internal state
        self.x = [None] * l # external input 
        self.f = [None] * l # forget gate
        self.i = [None] * l # input gate
        self.g = [None] * l # pre-activate state
        self.o = [None] * l # output gate
        self.h = [None] * l # actual output 
        y = [None] * l # your know, what every you want
        
        ct = self.c0 
        for t in range(l):
            xt = x[t]
            ft = utils.sigmoid(xt @ self.Uf + ct @ self.Wf + self.bf)
            it = utils.sigmoid(xt @ self.Ui + ct @ self.Wi + self.bi)
            gt = np.tanh(xt @ self.Uc + ct @ self.Wc + self.bc)
            ct = ft * ct + it * gt
            ot = utils.sigmoid(xt @ self.Uo + ct @ self.Wo + self.bo)
            ht = ot * np.tanh(ct)
            yt = ht @ self.V + self.by

            self.x[t] = xt
            self.f[t] = ft
            self.i[t] = it
            self.g[t] = gt
            self.c[t] = ct
            self.o[t] = ot
            self.h[t] = ht
            y[t] = yt
        return y

    def backward(self, dy):
        dUf = np.zeros_like(self.Uf)
        dUi = np.zeros_like(self.Ui)
        dUo = np.zeros_like(self.Uo)
        dUc = np.zeros_like(self.Uc)

        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWo = np.zeros_like(self.Wo)
        dWc = np.zeros_like(self.Wc)

        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbo = np.zeros_like(self.bo)
        dbc = np.zeros_like(self.bc)
        
        dV = np.zeros_like(self.V)
        dby = np.zeros_like(self.by)
        
        l = len(dy)
        for t in reversed(range(l)):
            # for one time point only    
            dUft = np.zeros_like(self.Uf)
            dUit = np.zeros_like(self.Ui)
            dUot = np.zeros_like(self.Uo)
            dUct = np.zeros_like(self.Uc)

            dWft = np.zeros_like(self.Wf)
            dWit = np.zeros_like(self.Wi)
            dWot = np.zeros_like(self.Wo)
            dWct = np.zeros_like(self.Wc)

            dbft = np.zeros_like(self.bf)
            dbit = np.zeros_like(self.bi)
            dbot = np.zeros_like(self.bo)
            dbct = np.zeros_like(self.bc)
            
            dVt = np.zeros_like(self.V)
            dbyt = np.zeros_like(self.by)

            dyt = dy[t]

            dht, dVt, dbyt = utils.backward(dyt, self.h[t], self.V)

            dot = dht * np.tanh(self.c[t])
            

            dct = dht * self.o[t] * (1 - self.c[t] ** 2)
            _, dWct, _ = utils.backward(dct, self.c[t -1], self.Wc[t]) 
            _, dUct, _ = utils.backward(dct, self.x[t], self.Uc[t])

            dft = dct * self.c[t - 1]
            _, dWft, _ = utils.backward(dft, self.c[t -1], self.Wf[t]) 
            _, dUft, _ = utils.backward(dft, self.x[t], self.Uf[t])

            dgt = dct * self.i[t]
            _, dWgt, _ = utils.backward(dgt, self.c[t -1], self.Wg[t]) 
            _, dUgt, _ = utils.backward(dgt, self.x[t], self.Ug[t])

            dit = dct * self.g[t]


            dht = dst * (1 - self.h[t] ** 2)
            dbht = dht
            for i in reversed(range(t)):
                dxi, dUi, _ = utils.backward(dht, self.x[i], self.U)
                if i > 0:
                    dsi, dWi, _ = utils.backward(dht, self.s[i - 1], self.W)
                    dht = np.clip((dht @ self.W.T) * (1 - self.h[i - 1] ** 2), -self.clip, self.clip)
                else:
                    dsi, dWi, _ = utils.backward(dht, self.s0, self.W)
                
                if np.mean(dht) < 0.001:
                    dht *= 10
                    #print('vanishing gradient')

                
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


        return (dV, dW, dU, dby, dbh)

    def learn(self, gradient):
        dVt, dWt, dUt, dbyt, dbht = gradient
        self.dV = utils.compute_momentum(self.dV, dVt, self.config)
        self.dW = utils.compute_momentum(self.dW, dWt, self.config)
        self.dU = utils.compute_momentum(self.dU, dUt, self.config)
        self.dby = utils.compute_momentum(self.dby, dbyt, self.config)
        self.dbh = utils.compute_momentum(self.dbh, dbht, self.config)

        self.V += self.dV
        self.W += self.dW
        self.U += self.dU
        self.by += self.dby
        self.bh += self.dbh
        

        #self.V -= self.config['step_size'] * dVt
        #self.W -= self.config['step_size'] * dWt
        #self.U -= self.config['step_size'] * dUt

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



    def translate(self, yhat, idx2char):
        return ''.join([idx2char[np.argmax(y)] for y in yhat])

    
    def fit(self, x, y, iter, char2idx, idx2char):
        l = len(x)
 

        for t in range(iter):
            self.s0 = np.zeros([1, self.dim_hidden])
            i = 0
            while i < l:
                xb = x[i:i + self.l]
                yb = y[i:i + self.l]
                yhat = self.forward(xb)
                loss, dy = compute_rnn_loss(yhat, yb)
                grad = self.backward(dy)
                self.learn(grad)
                self.s0 = self.s[-1]
                tt = self.translate(yhat, idx2char)
                print(t, loss, tt)
               
                i += self.l
            
            



