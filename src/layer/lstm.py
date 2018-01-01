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
        self.l = config['l']
        
    def accept(self, shape_in):      
        self.shape_in = shape_in 
        self.shape = self.shape_in
        self.dim_in = np.prod(self.shape_in, dtype=int)
        self.dim_out = np.prod(self.shape, dtype=int)
        
        # param 
        self.model = {
            'c0' : np.zeros([1, self.dim_hidden])
          , 'h0' : np.zeros([1, self.dim_hidden])
          , 'Uf' : np.random.randn(self.dim_in, self.dim_hidden) / np.sqrt(self.dim_in)
          , 'Ui' : np.random.randn(self.dim_in, self.dim_hidden) / np.sqrt(self.dim_in)
          , 'Uo' : np.random.randn(self.dim_in, self.dim_hidden) / np.sqrt(self.dim_in)
          , 'Ug' : np.random.randn(self.dim_in, self.dim_hidden) / np.sqrt(self.dim_in)
          , 'Wf' : np.random.randn(self.dim_hidden, self.dim_hidden) / np.sqrt(self.dim_hidden)
          , 'Wi' : np.random.randn(self.dim_hidden, self.dim_hidden) / np.sqrt(self.dim_hidden)
          , 'Wo' : np.random.randn(self.dim_hidden, self.dim_hidden) / np.sqrt(self.dim_hidden)
          , 'Wg' : np.random.randn(self.dim_hidden, self.dim_hidden) / np.sqrt(self.dim_hidden)
          , 'bf' : np.zeros((1, self.dim_hidden))
          , 'bi' : np.zeros((1, self.dim_hidden))
          , 'bo' : np.zeros((1, self.dim_hidden))
          , 'bg' : np.zeros((1, self.dim_hidden))
          , 'V'  : np.random.randn(self.dim_hidden, self.dim_out) / np.sqrt(self.dim_hidden)
          , 'by' : np.zeros((1, self.dim_out))
        }
        # cache
        self.cache = {
            'x' : []
          , 'f' : []
          , 'i' : []
          , 'g' : []
          , 'o' : []
          , 'c' : []
          , 'h' : []
        }
    
    def forward(self, xs, c, h):
        l = len(xs)
        y = [] # your know, what every you want
        

        m = self.model
        
        Wf, Wi, Wo, Wg = m['Wf'], m['Wi'], m['Wo'], m['Wg']
        Uf, Ui, Uo, Ug = m['Uf'], m['Ui'], m['Uo'], m['Ug']
        bf, bi, bo, bg = m['bf'], m['bi'], m['bo'], m['bg']
        V, by = m['V'], m['by']

        cache = self.cache
        for k, v in self.cache.items():
            v.clear()

        for t in range(l):
            x = xs[t]
            f = utils.sigmoid(x @ Uf + h @ Wf + bf)
            i = utils.sigmoid(x @ Ui + h @ Wi + bi)
            o = utils.sigmoid(x @ Uo + h @ Wo + bo)
            g = np.tanh(x @ Ug + h @ Wg + bg)
            c = f * c + i * g
            h = o * np.tanh(c)
            y.append(h @ V + by)
            cache['x'].append(x) 
            cache['f'].append(f)
            cache['i'].append(i)
            cache['o'].append(o)
            cache['g'].append(g)
            cache['c'].append(c)
            cache['h'].append(h)
        return y

    def backward(self, dys, dh_prev, dc_prev):
        d = {k:np.zeros_like(v) for k, v in self.model.items()}        
        l = len(dys)

        m = self.model
        Wf, Wi, Wo, Wg = m['Wf'], m['Wi'], m['Wo'], m['Wg']
        Uf, Ui, Uo, Ug = m['Uf'], m['Ui'], m['Uo'], m['Ug']
        bf, bi, bo, bg = m['bf'], m['bi'], m['bo'], m['bg']
        V, by = m['V'], m['by']

        cache = self.cache
        x, f, i, o, g, c, h = cache['x'], cache['f'], cache['i'], cache['o'], cache['g'], cache['c'], cache['h']

        for t in reversed(range(l)):
            # for one time point only    
            dt = {k:np.zeros_like(v) for k, v in self.model.items()}        
            
            dy = dys[t]
            dh, dV, dby = utils.backward(dy, h[t], V)
            dh += dh_prev
            
            do = dh * np.tanh(c[t]) 
            dbo = do
            dho, dWo, _ = utils.backward(do, h[t - 1], Wo) 
            _, dUo, _ = utils.backward(do, x[t], Uo)

            dc = dh * o[t] * (1 - c[t] ** 2)

            dc += dc_prev

            df = dc * g[t]
            dbf = df
            dhf, dWf, _ = utils.backward(df, h[t -1], Wf) 
            _, dUf, _ = utils.backward(df, x[t], Uf)

            dg = dc * i[t]
            dbg = dg
            dhg, dWg, _ = utils.backward(dg, h[t -1], Wg) 
            _, dUg, _ = utils.backward(dg, x[t], Ug)

            di = dc * g[t]
            dbi = di
            dhi, dWi, _ = utils.backward(di, h[t -1], Wi) 
            _, dUi, _ = utils.backward(di, x[t], Ui)

            dh_prev = dho + dhf + dhi + dhg
            dc_prev = dc * f[t]

            d['Uf'] += dUf
            d['Ui'] += dUi
            d['Ug'] += dUg
            d['Uo'] += dUo

            d['Wf'] += dWf
            d['Wi'] += dWi
            d['Wg'] += dWg
            d['Wo'] += dWo

            d['bf'] += dbf
            d['bi'] += dbi
            d['bg'] += dbg
            d['bo'] += dbo

            d['V'] += dV
            d['by'] += dby

        return d

    def learn(self, grad):
        for k, v in grad.items():
            self.model[k] -= 0.01 * v
         

    def sample(self, char, l, char2idx, idx2char):
        y = [None] * (l + 1)
        x = np.zeros([1, self.dim_in])
        x[0][char2idx[char]] = 1.0
        h = self.model['h0']
        c = self.model['c0']
        y[0] = char
        m = self.model
        Wf, Wi, Wo, Wg = m['Wf'], m['Wi'], m['Wo'], m['Wg']
        Uf, Ui, Uo, Ug = m['Uf'], m['Ui'], m['Uo'], m['Ug']
        bf, bi, bo, bg = m['bf'], m['bi'], m['bo'], m['bg']
        V, by = m['V'], m['by']

        for t in range(l):
            f = utils.sigmoid(x @ Uf + h @ Wf + bf)
            i = utils.sigmoid(x @ Ui + h @ Wi + bi)
            o = utils.sigmoid(x @ Uo + h @ Wo + bo)
            g = np.tanh(x @ Ug + h @ Wg + bg)
            c = f * c + i * g
            h = o * np.tanh(c)
            yhat = h @ V + by
            y[t + 1] = idx2char[np.argmax(yhat)]

        print(y)
        return ''.join(y)



    def translate(self, yhat, idx2char):
        return ''.join([idx2char[np.argmax(y)] for y in yhat])

    
    def fit(self, x, y, l, iter, char2idx, idx2char):
        lx = len(x)
        for t in range(iter):
            i = 0
            ht = self.model['h0']
            ct = self.model['c0']
            dh_prev = self.model['h0']
            
            dc_prev = self.model['c0']
            ##if t % 100 == 0:
             #   self.config['step_size'] *= 0.8

            while i < lx:
                e = min(lx, i + l)
                xb = x[i:e]
                yb = y[i:e]
                yhat = self.forward(xb, ct, ht)
                loss, dy = utils.compute_rnn_loss(yhat, yb)
                grad = self.backward(dy, dh_prev, dc_prev)
                self.learn(grad)
                ht = self.cache['h'][-1]
                ct = self.cache['c'][-1]
                tt = self.translate(yhat, idx2char)
                print(t, loss, tt)
                i += l
            
            
            



