"""
fully connected layer class
"""

import numpy as np
from layer import Layer 
import utils 

class FullyConnectedLayer(Layer):
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
      g_y  | (T, m)           | gradient on output                             
      g_x  | (T, n)           | gradient on input                              
      g_w  | (n, m)           | gradient on weight
      g_b  | (1, m)           | gradient on bias                  
    ============================================================================
    '''
    def __init__(self, output):
        super(FullyConnectedLayer, self).__init__()
        self.type = 'FC Layer'
        self.output = np.prod(output, dtype=int)
    
    def accept(self, src):
        self.input = np.prod(src, dtype=int)

        if self.input <= 0:
            return False

        self.w = np.random.normal(0, 1, [self.input, self.output])
        self.b = np.random.normal(0, 1, [1, self.output])
        self.src = [self.input]
        self.shape = [self.output]
        self.x = None
        return True
        
    def forward(self, x):
        y = utils.forward(x, self.w, self.b) 
        self.x = x
        return y

    def backward(self, dy):
        dx, dw, db = utils.backward(dy, self.x, self.w)
        
        self.w -= dw
        self.b -= db

        return dx