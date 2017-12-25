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
        self.dw = None
        self.db = None
        return True
        
    def forward(self, x):
        self.x = x
        return utils.forward(x, self.w, self.b) 
        
    def backward(self, dy):
        N = dy.shape[0]
        dx, dw, db = utils.backward(dy, self.x, self.w)
        self.x = None

        self.dw = dw / N
        self.db = db / N
        return dx
    
    def learn(self, param):
        self.w -= self.dw * param['step_size']
        self.b -= self.db * param['step_size'] 