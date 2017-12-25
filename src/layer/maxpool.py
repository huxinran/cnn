from layer import Layer 
import numpy as np
import utils


class MaxPoolLayer(Layer):
    def __init__(self, kernel_shape=(2, 2)):
        super(MaxPoolLayer, self).__init__()
        self.type = 'MaxPool'
        self.kernel_shape = kernel_shape 
        self.pad = (0, 0)
        self.stride = self.kernel_shape

    def accept(self, input_shape):
        if input_shape[1] % self.kernel_shape[0] != 0:
            return False
        
        if input_shape[2] % self.kernel_shape[1] != 0:
            return False

        self.input_shape = input_shape

        self.shape = np.array((input_shape[0], 
                               input_shape[1] / self.kernel_shape[0], 
                               input_shape[2] / self.kernel_shape[1]), dtype=int)
        
        self.indice = utils.flatten_index(self.input_shape, 
                                          self.kernel_shape, 
                                          self.pad, 
                                          self.stride)

        self.d_in = np.prod(self.input_shape, dtype=int)
        self.d_out = np.prod(self.shape, dtype=int)
        self.d_kernel = np.prod(self.kernel_shape, dtype=int)
        self.max_indice = None
        return True

    def forward(self, x):
        N = x.shape[0]
        
        y = np.zeros([N, self.d_out])
        
        self.max_indice = np.zeros([N, self.d_out], dtype=int)
        
        for i in range(N):
            flat_xi = utils.flatten(x[i,].reshape(self.input_shape), 
                                   self.input_shape, 
                                   self.kernel_shape, 
                                   self.pad, 
                                   self.stride, 
                                   self.indice).reshape(-1, self.d_kernel)

            self.max_indice[i, ] = np.argmax(flat_xi, axis=1)

            y[i, ] = flat_xi[np.arange(self.d_out), self.max_indice[i, ]].reshape(-1, self.input_shape[0]).T.ravel()
        return y

    def backward(self, dy):
        N = dy.shape[0]
        
        dx = np.zeros([N, self.d_in])

        for i in range(N):
            dflat_x = np.zeros([self.d_out, self.d_kernel])

            dflat_x[np.arange(self.d_out), self.max_indice[i,]] = dy[i,].reshape(self.shape).transpose([1, 2, 0]).ravel()
            
            dx[i, ] = utils.unflatten(dflat_x, 
                                      self.input_shape, 
                                      self.kernel_shape, 
                                      self.pad, 
                                      self.stride, 
                                      self.indice).ravel()
        return dx
    
    def learn(self, config):
        pass