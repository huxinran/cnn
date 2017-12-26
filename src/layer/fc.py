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

    
    def __init__(self, shape):
        super(FullyConnectedLayer, self).__init__()
        self.type = 'FullyConnected'
        self.shape = shape
        self.dim_out = np.prod(self.shape, dtype=int)

    def accept(self, shape_in):
        self.shape_in = shape_in
        self.dim_in = np.prod(self.shape_in, dtype=int)
        
        # params
        self.w = np.random.normal(0, 1.0 / np.sqrt(self.dim_in), [self.dim_in, self.dim_out])
        self.b = np.random.normal(0, 1.0 / np.sqrt(self.dim_in), [1, self.dim_out])
        
        # cache
        self.x = None
        self.dw_m = np.zeros([self.dim_in, self.dim_out])
        self.db_m = np.zeros([1, self.dim_out])
        self.dw = None
        self.db = None
        return True
        
    def forward(self, x):
        self.x = x
        return utils.forward(x, self.w, self.b) 
        
    def backward(self, dy):
        N = dy.shape[0]
        dx, dw, db = utils.backward(dy, self.x, self.w)
        self.dw = dw
        self.db = db
        return dx
    
    def update(self, config):
        self.dw_m = utils.compute_momentum(self.dw_m, self.dw, config)    
        self.dw_b = utils.compute_momentum(self.db_m, self.db, config)
        self.w += self.dw_m
        self.b += self.db_m