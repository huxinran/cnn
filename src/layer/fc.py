'''
Fully Connected Layer Class represents a general function f(x, w) = y
it provides 3 utility functions


List of Variable
============================================================================
    Name | Type             | Explanation                                    
============================================================================
    n    | int              | dimension of input                             
    m    | int              | dimension of output                            
    T    | int              | num of inputs
---------------------------------------------------------------------------
    x    | (T, n)           | input                                 
    w    | (n, m)           | weight
    b    | (1, m)           | bias                        
    y    | (T, m)           | output                   
----------------------------------------------------------------------------
    dy   | (T, m)           | gradient on output                             
    dx   | (T, n)           | gradient on input                              
    dw   | (n, m)           | gradient on weight
    db   | (1, m)           | gradient on bias                  
============================================================================
'''

from layer import Layer 
import numpy as np
import utils 

class FullyConnectedLayer(Layer):
    '''
    Fully Connected Layer Class represents a general function f(x, w) = y
    it provides 3 utility functions
    '''
    def __init__(self, config):
        super(FullyConnectedLayer, self).__init__(config)
        self.config = config
        self.type = 'FullyConnected'
        self.shape = config['shape']
        self.dim = np.prod(self.shape, dtype=int)

    def accept(self, src_shape):
        self.shape_in = src_shape
        self.dim_in = np.prod(self.shape_in, dtype=int)
        
        # params
        self.param = {
            'w' : np.random.randn(self.dim_in, self.dim) / np.sqrt(self.dim_in)  
          , 'b' : np.ones(1, self.dim) * 0.1
        }    

        # cache
        return True
        
    def forward(self, x):
        cache = {
            'x' : x
        }
        y = utils.forward(x, self.param['w'], self.param['b']) 
        return y, cache

    def backward(self, dy):
        dx, dw, db = utils.backward(dy, self.cache['x'], self.param['w'])
        dparam = {
            'w' : dw
          , 'b' : db
        }        
        return dx, dparam
    
    def learn(self, dparam):
        self.dw_m = utils.compute_momentum(self.dw_m, dparam['w'], config)    
        self.dw_b = utils.compute_momentum(self.db_m, dparam['b'], config)
        self.w += self.dw_m
        self.b += self.db_m