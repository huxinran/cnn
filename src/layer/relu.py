from layer import Layer 
import numpy as np

class ReluLayer(Layer):
    def __init__(self):
        super(ReluLayer, self).__init__()
        self.type = 'Relu'
    
    def accept(self, shape_in):
        self.shape_in = shape_in
        self.shape = shape_in
        self.dim_in = np.prod(self.shape_in, dtype=int)
        self.dim_out = np.prod(self.shape, dtype=int)
        return True

    def forward(self, x):
        self.x = np.maximum(0, x)
        return self.x
    
    def backward(self, dy):
        return dy * (1 * self.x > 0)
    
    def learn(self, config):
        return