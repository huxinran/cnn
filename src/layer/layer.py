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
        try to accept a src layer with a specific shape

        if this layer can accept the src as input, 
        return True, otherwise return false
        '''
        raise Exception('accept is not implemented')

    def forward(self, x):
        '''
        given data x, return output and cache
        '''
        raise Exception('forward is not implemented')

    def backward(self, dy):
        '''
        given dy, compute d_param and d_x
        '''
        raise Exception('backward is not implemented')

    def update(self, d_param):
        '''
        
        '''
        raise Exception('learn is not implmented')