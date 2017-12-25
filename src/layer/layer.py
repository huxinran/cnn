class Layer:
    def __init__(self):
        '''
        five common parameters of all types of layer
        '''
        self.type = 'Unknown'
        self.shape_in = []
        self.shape = []
        self.dim_in = 0
        self.dim_out = 0
        
    def __repr__(self):
        return '{0} Layer | {1} => {2}'.format(self.type, self.shape_in, self.shape)

    # each layer type, please implement the following
    def accept(self, src):
        '''
        given a src with is a shape 
        if you can take src as input, update shape as your output 
        return True
        if not return false
        '''
        raise Exception('accept is not implemented')

    def forward(self, x):
        '''
        given x, 
        (1) use self.param to compute a y,
        (2) set self.cahce if needed
        (3) return y
        '''
        raise Exception('forward is not implemented')

    def backward(self, dy):
        '''
        given dy,
        (1) use self.cache to comput dx
        (2) set self.param accordingly
        (3) return dx
        '''
        raise Exception('backward is not implemented')

    def learn(self, config):
        '''
        call after backward
        '''
        raise Exception('learn is not implmented')