from layer import Layer 
import numpy as np


class MaxPoolLayer(Layer):
    def __init__(self, kernel_h=2, kernel_w=2):
        super(MaxPoolLayer, self).__init__()
        self.type = 'MaxPool'
        pass
    
    def accept(self, src):
        self.src = np.array(src)
        self.shape = np.array([src[0], src[1] / kernel_h, src[2] / kernel_w])
        self.input = np.prod(self.src, dtype=int)
        self.output = np.prod(self.shape, dtype=int)
        pass

    def forward(self, x):
        pass

    def backward(self, dy):
        pass