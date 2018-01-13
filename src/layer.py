class Layer:
    def __init__(self, config):
        '''
        five common parameters of all types of layer
        '''
        self.config = config
        
        self.type = 'Unknown'
        
        # input shape and dimension
        self.shape_in = []
        self.dim_in = 0
        
        # output shape and dimension
        self.shape = []
        self.dim = 0
        
        # model param
        self.param = {}

        # storing all intermediate
        self.cache = {}

    def __repr__(self):
        return '{0} Layer | {1} => {2}'.format(self.type, self.shape_in, self.shape)

    # each layer type, please implement the following
    def accept(self, src):
        '''
        try to accecpt a src layer
        '''
        raise Exception('accept is not implemented')

    def forward(self, x):
        '''
        given x, return output and cache
        '''
        raise Exception('forward is not implemented')

    def backward(self, dy):
        '''
        given dy, return dparam and dx
        '''
        raise Exception('backward is not implemented')

    def learn(self, dparam):
        '''
        call after backward
        '''
        raise Exception('learn is not implmented')