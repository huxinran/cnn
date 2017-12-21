"""
fully connected layer class
"""
import numpy as np
class FullyConnectedLayer:
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
    @staticmethod
    def init_weight(n, m):
        '''
        init weight (with bias)
        '''
        sd = 1 / np.sqrt(n)
        return np.random.normal(0, sd, [n, m]), np.random.normal(0, sd, [1, m])

    @staticmethod
    def fwd(x, w, b):
        '''
        y = x * w + b
        '''
        return np.dot(x, w) + b

    @staticmethod
    def bwd(dy, x, w):
        '''
        dx = dy * w
        dw = x.T * dy
        db = 1.T * dy
        '''
        dx = np.dot(g_y, w.T)
        dw = np.dot(x.T, dy)
        db = np.sum(dy, axis = 0)
        return dx, dw, db




