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

    
    def __init__(self, output_shape):
        super(FullyConnectedLayer, self).__init__()
        self.type = 'FullyConnected'
        self.shape = np.array(output_shape)
    
    def accept(self, input_shape):
        self.input_shape = np.array(input_shape)

        # params
        d_in = np.prod(self.input_shape, dtype=int)
        d_out = np.prod(self.shape, dtype=int)
        self.w = np.random.normal(0, 1, [d_in, d_out])
        self.b = np.random.normal(0, 1, [1, d_out])
        
        # cache
        self.x = None
        self.dw = None
        self.db = None
        return True
        
    def forward(self, x):
        self.x = x
        return utils.forward(x, self.w, self.b) 
        
    def backward(self, dy):
        dx, dw, db = utils.backward(dy, self.x, self.w)
        self.x = None
        self.dw = dw
        self.db = db
        return dx
    
    def learn(self, param):
        self.w -= self.dw * param['step_size']
        self.b -= self.db * param['step_size'] 