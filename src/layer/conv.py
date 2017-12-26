from layer import Layer 
import utils
import numpy as np

class ConvLayer(Layer):
    def __init__(self, shape_k, dout, pad=0, stride=1):        
        super(ConvLayer, self).__init__()
        self.type = 'Convolution'
        self.shape_k = shape_k
        self.hk, self.wk = self.shape_k
        self.dout = dout
        self.pad = (pad, pad)
        self.stride = (stride, stride)

    def accept(self, shape_in):
        if (shape_in[1] + 2 * self.pad[0] - self.hk + 1) % self.stride[0] != 0:
            return False
        
        if (shape_in[2] + 2 * self.pad[1] - self.wk + 1) % self.stride[1] != 0:
            return False

        self.shape_in = shape_in        
        self.dim_in = np.prod(self.shape_in, dtype=int)
        self.din, self.hin, self.win = self.shape_in
        self.dk = self.din
        self.hpos = utils.get_pos(self.hin, self.hk, self.pad[0], self.stride[0])
        self.wpos = utils.get_pos(self.win, self.wk, self.pad[1], self.stride[1])
        self.hout = self.hpos.size
        self.wout = self.wpos.size
        self.shape = [self.dout, self.hout, self.wout]
        self.dim_out = np.prod(self.shape, dtype=int)
        self.indice = utils.flatten_index(self.shape_in, 
                                          self.shape_k, 
                                          self.pad, 
                                          self.stride)
        self.dim_k = self.dk * self.hk * self.wk
        
        # params
        self.w = np.random.normal(0, 1.0 / np.sqrt(self.dim_k), [self.dim_k, self.dout])
        self.b = np.random.normal(0, 1.0 / np.sqrt(self.dim_k), [1, self.dout])
        
        # cache
        self.fx = None
        self.dw_m = np.zeros([self.dim_k, self.dout])
        self.db_m = np.zeros([1, self.dout])
        self.dw = None
        self.db = None
        
        return True 

    def forward(self, x):
        '''
        '''
        N = x.shape[0]
        y = np.zeros((N, self.dim_out)) 

        self.fx = np.zeros((N, self.hout * self.wout, self.dim_k))
        
        for i in range(N):
            self.fx[i,] = utils.flatten(x[i, :].reshape(self.shape_in),
                                        self.shape_in, 
                                        self.shape_k, 
                                        self.pad,
                                        self.stride, 
                                        self.indice).reshape(self.hout * self.wout, -1) 
            y[i,] = utils.forward(self.fx[i,], self.w, self.b).T.ravel()
        return y

    def backward(self, dy):
        N = dy.shape[0]
        dx = np.zeros([N, self.dim_in])
        dw = np.zeros([self.dim_k, self.dout])
        db = np.zeros([1, self.dout])

        for i in range(N):
            dyi = dy[i, :].reshape(self.dout, -1).T

            dfxi, dwi, dbi = utils.backward(dyi, self.fx[i, ], self.w)

            dx[i,] = utils.unflatten(dfxi, 
                                     self.shape_in,
                                     self.shape_k,
                                     self.pad,
                                     self.stride,
                                     self.indice).ravel() 
            dw += dwi
            db += dbi

        self.dw = dw
        self.db = db 
        return dx
    
    def update(self, config):
        self.dw_m = utils.compute_momentum(self.dw_m, self.dw, config)
        self.db_m = utils.compute_momentum(self.db_m, self.db, config)
        self.w += self.dw_m
        self.b += self.db_m