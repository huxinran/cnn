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




