from layer import Layer 
import numpy as np

class ReluLayer(Layer):
    def __init__(self):
        super(ReluLayer, self).__init__()
        self.type = 'Relu'
    
    def accept(self, src):
        self.src = np.array(src)
        self.input_shape = np.array(src)
        self.shape = np.array(src)
        return True

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, dy):
        return dy.reshape(self.x.shape) * (1 * self.x > 0)
    
    def learn(self, config):
        return