<<<<<<< HEAD
class Neuron:
    def __init__(self, inputDim, activateFunc):
        self.inputDim = inputDim
        self.activateFunc = activateFunc
        self.weight = np.zeros(inputDim + 1, 1)
        self.value = 0.0

    def forward(input):
        self.value = max(0, np.inner(input, self.weight))
        return self.value
    
    def backward(gradient):
        

    
=======
import numpy as np

class Neuron:
    def __init__(self, d):
        '''
        init a neuron that takes a d-dimensional input
        w is a dx1 weight array, init with gaussian random number
        b is a 1x1 bias, init with gaussian random number
        '''
        self.d = d
        self.w = np.random.normal(0, 1, self.d)
        self.b = np.random.normal(0, 1, 1)

    def __repr__(self):
        return 'w={0}, b={1}'.format(self.w, self.b)

    def forward(self, x):
        '''
        given a input x, 
        return wx + b
        '''     
        return np.inner(self.w, x) + self.b
     
    def backward(self, x):
        '''
        given x, 
        return dx, dw, db 
        '''
        return self.w, x, np.array([1])
>>>>>>> f91831aaff2d941e5e60b416fd0c9ae05f8ea307
