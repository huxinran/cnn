from layer import Layer 
import numpy as np

class ReluLayer(Layer):
    def __init__(self, config):
        super(ReluLayer, self).__init__(config)
        self.type = 'Relu'
        self.config = config
        
    def accept(self, shape_in):
        self.shape_in = shape_in
        self.shape = shape_in
        self.dim_in = np.prod(self.shape_in, dtype=int)
        self.dim = np.prod(self.shape, dtype=int)
        return True

    def forward(self, x):
        cache = {
            'mask' : np.maximum(0, x) > 0
        }
        y = x * cache['mask']
        return y, cache
    
    def backward(self, dy):
        dx = dy * self.cache['mask']
        dparam = None
        return dx, dparam
    
    def learn(self, dparam, step_size):
        return