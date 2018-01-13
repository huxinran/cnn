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
        super(FullyConnectedLayer, self).__init__()
        self.type = 'FullyConnected'
        self.config = config
        self.shape = config['shape']
        self.dim_out = np.prod(self.shape, dtype=int)

    def accept(self, shape_in):
        self.shape_in = shape_in
        self.dim_in = np.prod(self.shape_in, dtype=int)
        
        # params
        self.param = {
            'w' : np.random.randn(self.dim_in, self.dim_out) / np.sqrt(self.dim_in)
          , 'b' : np.random.ones(1, self.dim_out) * 0.1
        }

        # cache
        self.cache = {}
        
        self.dw_m = np.zeros([self.dim_in, self.dim_out])
        self.db_m = np.zeros([1, self.dim_out])
        self.dw = None
        self.db = None
        return True
        
    def forward(self, x):
        cache = {
            'x' : x
        }
        y = utils.forward(x, self.params['w'], self.params['b'])
        return y, cache
        
    def backward(self, dy):
        dx, dw, db = utils.backward(dy, self.cache['x'], self.param['w'])
        self.dw = dw
        self.db = db
        dparam = {
            'w' : dw
          , 'b' : db
        }
        return dx, dparam
    
    def update(self, config):
        self.dw_m = utils.compute_momentum(self.dw_m, self.dw, config)    
        self.dw_b = utils.compute_momentum(self.db_m, self.db, config)
        self.w += self.dw_m
        self.b += self.db_m