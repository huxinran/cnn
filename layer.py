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
    | Name | Type             | Explanation                                    |
    ============================================================================
    | din  | int              | dimension of input                             |
    | dout | int              | dimension of output                            |
    ----------------------------------------------------------------------------
    | x    | (N, din)         | input variable                                 |
    | w    | (din + 1 , dout) | weight (+1 due to bias)                        |
    | y    | (N , dout)       | output variable y = x * w                      |
    |--------------------------------------------------------------------------|
    | d_y  | (N , dout)       | gradient on output                             |
    | d_x  | (N , din)        | gradient on input   d_x = d_y * d_w[1:,]       |
    | d_w  | (din + 1 , dout) | gradient on weight  d_w = t([1, d_x]) * d_y    |
    ============================================================================

    List of Instance Method
    ============================================================================
    |name     | type                      | detail                             |
    ============================================================================
    |fwd      | (x, w) => y               | forward feed                       |
    |bwd      | (d_y, x, w) => (d_x, d_w) | back prop                          |
    ============================================================================
    '''
    @staticmethod
    def init_weight(dim_in, dim_out):
        '''
        init weight (with bias)
        '''
        return np.random.normal(0, 1 / np.sqrt(dim_in), [dim_in + 1, dim_out])

    @staticmethod
    def fwd(x, w):
        '''
        add a bias term to x, then compute output
        '''
        return np.dot(np.c_[np.ones(x.shape[0]), x], w)

    @staticmethod
    def bwd(d_y, x, w):
        '''
        given gradient on output, value of x and w
        computer gradient on w (including bias) and gradient on x
        '''
        d_x = np.dot(d_y, w[1:,].T)
        d_w = np.dot(np.r_[np.ones([1, d_y.shape[0]]), x.T], d_y)
        return d_x, d_w
