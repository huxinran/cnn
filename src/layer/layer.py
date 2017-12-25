import numpy as np

class Layer:
    def __init__(self):
        self.type = 'Unknown'
        self.ready = False
        self.src = None
        self.shape = None
        self.cache = None
        self.param = None
        
    def __repr__(self):
        return '{0} Layer | from {1} to {2}'.format(self.type, self.src, self.shape)

    
    # each layer type, please implement the following
    def accept(self, src):
        '''
        given a src with is a shape 
        if you can take src as input, update shape as your output 
        return True
        if not return false
        '''
        return True

    def feed_forward(self, x):
        '''
        given x, 
        (1) use self.param to compute a y,
        (2) set self.cahce if needed
        (3) return y
        '''
        return None

    def feed_backward(self, dy):
        '''
        given dy,
        (1) use self.cache to comput dx
        (2) set self.param accordingly
        (3) return dx
        '''
        return None
