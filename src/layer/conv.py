from layer import Layer 
import utils
import numpy as np
import matplotlib.pyplot as plt

class ConvLayer(Layer):
    def __init__(self, config):        
        super(ConvLayer, self).__init__(config)
        self.type = 'Conv'
        self.config = config
        
        self.shape_k = config['kernel_shape']
        self.dout = config['output_depth']
        self.pad = config['pad']
        self.stride = config['stride']
        
        self.hk, self.wk = self.shape_k
        
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
        self.dim = np.prod(self.shape, dtype=int)
        self.indice = utils.flatten_index(self.shape_in, 
                                          self.shape_k, 
                                          self.pad, 
                                          self.stride)
        self.dim_k = self.dk * self.hk * self.wk
        
        # params
        self.param = {
            'w' : np.random.randn(self.dim_k, self.dout) / np.sqrt(self.dim_k)
          , 'b' : np.ones([1, self.dout]) * 0.1
        }

        self.paramSum = {
            'w' : np.ones_like(self.param['w']) * 1e-8
          , 'b' : np.ones_like(self.param['b']) * 1e-8
        }
        
        # cache
        self.cache = {}
        self.fx = None
        self.dw = np.zeros([self.dim_k, self.dout])
        self.db = np.zeros([1, self.dout])
        self.dw_cache = None
        self.db_cache = None
        return True 

    def forward(self, x):
        '''
        '''
        N = x.shape[0]
        y = np.zeros((N, self.dim)) 

        cache = {
            'flatten_x' : np.zeros((N, self.hout * self.wout, self.dim_k))
        }
        
        for i in range(N):
            cache['flatten_x'][i,] = utils.flatten(x[i, :].reshape(self.shape_in),
                                          self.shape_in, 
                                          self.shape_k, 
                                          self.pad,
                                          self.stride, 
                                          self.indice).reshape(self.hout * self.wout, -1) 
            y[i,] = utils.forward(cache['flatten_x'][i,], self.param['w'], self.param['b']).T.ravel()
        return y, cache

    def backward(self, dy):
        N = dy.shape[0]
        dx = np.zeros([N, self.dim_in])
        
        dparam = {
            'w' : np.zeros([self.dim_k, self.dout])
          , 'b' : np.zeros([1, self.dout])
        }
        
        for i in range(N):
            dyi = dy[i, :].reshape(self.dout, -1).T
            fxi = self.cache['flatten_x'][i, ]
            
            dfxi, dwi, dbi = utils.backward(dyi, fxi, self.param['w'])

            dx[i,] = utils.unflatten(dfxi, 
                                     self.shape_in,
                                     self.shape_k,
                                     self.pad,
                                     self.stride,
                                     self.indice).ravel() 
            dparam['w'] += dwi
            dparam['b'] += dbi

        dparam['w'] /= N
        dparam['b'] /= N

        return dx, dparam
    
    def learn(self, dparam, step_size):
        utils.adam(self.param, self.paramSum, dparam, step_size)