from layer import Layer 
import numpy as np
import utils


class MaxPoolLayer(Layer):
    def __init__(self, config):
        super(MaxPoolLayer, self).__init__(config)
        self.type = 'MaxPool'
        self.shape_k = config['shape_k'] 
        self.pad = (0, 0)
        self.stride = self.shape_k

    def accept(self, shape_in):
        if shape_in[1] % self.shape_k[0] != 0:
            return False
        
        if shape_in[2] % self.shape_k[1] != 0:
            return False

        self.shape_in = shape_in

        self.shape = np.array((shape_in[0], 
                               shape_in[1] / self.shape_k[0], 
                               shape_in[2] / self.shape_k[1]), dtype=int)
        
        self.indice = utils.flatten_index(self.shape_in, 
                                          self.shape_k, 
                                          self.pad, 
                                          self.stride)

        self.dim_in = np.prod(self.shape_in, dtype=int)
        self.dim_out = np.prod(self.shape, dtype=int)
        self.dim_k = np.prod(self.shape_k, dtype=int)
        
        # cache
        self.max_indice = None
        return True

    def forward(self, x):
        N = x.shape[0]
        
        y = np.zeros([N, self.dim_out])
        
        self.max_indice = np.zeros([N, self.dim_out], dtype=int)
        
        for i in range(N):
            fxi = utils.flatten(x[i,].reshape(self.shape_in), 
                                self.shape_in, 
                                self.shape_k, 
                                self.pad, 
                                self.stride, 
                                self.indice).reshape(-1, self.dim_k)

            self.max_indice[i, ] = np.argmax(fxi, axis=1)

            y[i, ] = fxi[np.arange(self.dim_out), self.max_indice[i, ]].reshape(-1, self.shape_in[0]).T.ravel()
        return y

    def backward(self, dy):
        N = dy.shape[0]
        
        dx = np.zeros([N, self.dim_in])

        for i in range(N):
            dfxi = np.zeros([self.dim_out, self.dim_k])

            dfxi[np.arange(self.dim_out), self.max_indice[i,]] = dy[i,].reshape(self.shape).transpose([1, 2, 0]).ravel()
            
            dx[i, ] = utils.unflatten(dfxi, 
                                      self.shape_in, 
                                      self.shape_k, 
                                      self.pad, 
                                      self.stride, 
                                      self.indice).ravel()
        return dx
    
    def learn(self, config):
        pass