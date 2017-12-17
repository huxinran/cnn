import numpy as np 

class FullyConnectedLayer:
    '''
    Fully Connected Layer Class
    
    represents a function f(x, w) = y
    convert an input matrix, x into an output matrix, y


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
    |---------------------------------------------------------------------------        
    | dy   | (N , dout)       | gradient on output                             |
    | dx   | (N , din)        | gradient on input   dx = dy * dw[1:,]          |
    | dw   | (din + 1 , dout) | gradient on weight  dw = t([1, dx]) * dy       |
    ============================================================================

    List of Instance Method
    ============================================================================
    |name     | type                   | detail                                |
    ============================================================================
    |forward  | (x, w) => y            | forward feed                          |
    |backward | (x, w, dy) => (dx, dw) | back prop                             |
    ============================================================================
    '''

    def initWeight(din, dout):
        return np.random.normal(0, 1, [din + 1, dout])
        
    def predict(x, w):
        return np.dot(np.c_[np.ones(x.shape[0]), x], w)
        
    def gradient(dy, x, w):
        dx = np.dot(dy, w[1:,].T)
        dw = np.dot(np.r_[np.ones([1, dy.shape[0]]), x.T], dy)
        return dx, dw