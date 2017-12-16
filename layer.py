import numpy as np 

class FullyConnectedLayer:
    '''
    Fully Connected Layer Class
    
    (1) convert an input matrix, x into an output matrix, y
    (2) stores weights as well as corresponding temporary variables

    List of Instance Variables 
    ============================================================================
    |name | type             | detail                                          |
    ============================================================================
    |din  | int              | input dimension                                 |
    |dout | int              | output dimension                                |
    ----------------------------------------------------------------------------
    |x    | (1, din)         | input                                           |
    |w    | (din + 1 , dout) | weight                                          |
    |y    | (1 , dout)       | output                                          |
    |---------------------------------------------------------------------------        
    |dx   | (1 , din)        | input gradient                                  |
    |dw   | (din + 1 , dout) | weight gradient                                 |
    |dy   | (1 , dout)       | output gradient                                 |
    ============================================================================



    '''
    def __init__(self, din, dout):
        self.din = din
        self.dout = dout
        
        self.x = np.empty([1, din])
        self.w = np.empty([din + 1, dout])
        self.y = np.empty([1, dout])

        self.dx = np.empty([1, din])
        self.dw = np.empty([din + 1, dout])
        self.dy = np.empty([1, dout])

        self.w = np.random.normal(0, 1, [din + 1, dout])

    def __repr__(self):
        return 'Fully Connected Layer of shape {0.din} to {0.dout}'.format(self)

    def forward(self, x, w):
        return np.dot(np.append(1, x).reshape(1, -1), w)
        
    def backward(self, x, w, dy):
        dx = np.dot(dy, w[1:,].T)
        dw = np.outer(np.append(1, x).reshape(1, -1), dy)
        return dx, dw