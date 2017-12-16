import numpy as np

class Loss:
    def __init__(self):
        self.p = None

    def forward(self, s, y):
        p = np.exp(s)
        self.p = p / p.sum()
        self.y = y
        return -np.log(self.p[self.y])

    def backward(self, dz):
        g = self.p
        g[self.y] -= 1
        return dz * g