import numpy as np 

class FullyConnectedLayer:
    def __init__(self, din, dout):
        self.x = np.zeros(din)
        self.w = np.random.normal(0, 1, [din + 1, dout])
        
    def __repr__(self):
        return 'Fully Connected layer | w={0}'.format(self.w)

    def forward(self, x, w):
        return np.dot(np.append(1, x), w)
        
    def backward(self, x, w, dy):
        dx = np.dot(dy, w[1:,].T)
        dw = np.outer(np.append(1, x), dy)
        return dx, dw