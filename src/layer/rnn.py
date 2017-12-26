from layer import Layer
import utils


class RNNLayer(Layer):
    def __init__(self, config):
        super(RNNLayer, self).__init__()
        self.type = 'RNN'
        self.config = config
        self.dim_hidden = config['dim_hidden']
        
    def accept(self, shape_in):      
        self.shape_in = shape_in 
        self.shape = self.shape
        self.dim_in = np.prod(self.shape_in, dtype=int)
        self.dim_out = np.prod(self.shape, dtype=int)

        # param 
        self.wx = np.random.normal(0, 1, [self.dim_in, self.dim_hidden])
        self.wh = np.random.normal(0, 1, [self.dim_hidden, self.dim_hidden])
        self.wy = np.random.normal(0, 1, [self.dim_hiddem, self.dout])
        self.bx = np.random.normal(0, 1, [1, self.dim_hidden]) 
        self.bh = np.random.normal(0, 1, [1, self.dim_hidden])
        self.h = np.zeros([1, self.dim_hidden])

        #cache 
        self.x = np.zeros([1, self.dim_in])

    
    def forward(self, x_seq):
        self.x_seq = x_seq
        for i in range(len(x_seq)):
            xi = x_seq[i]
            
            print(i)

            self.h = self.h.dot(self.wh)
            y = self.x.dot(self.wx) + self.h
            return y

    def backward(self, dy, dh):
