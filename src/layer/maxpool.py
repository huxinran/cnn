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
        
        self.indice = utils.flatten_index(self.input_shape, self.kernel_shape, (0, 0), self.kernel_shape)
        return True

    def forward(self, x):
        N = x.shape[0]
        y = np.zeros([N, self.shape[0] * self.shape[1] * self.shape[2]])
        self.max_index = np.zeros([N, self.shape[0] * self.shape[1] * self.shape[2]], dtype=int)
        for i in range(N):
            flat_x = utils.flatten(x[i,].reshape(self.input_shape), self.input_shape, self.kernel_shape, self.pad, self.stride)
            flat_x = flat_x.reshape(-1, np.prod(self.kernel_shape, dtype=int))
            max_index = np.argmax(flat_x, axis=1)
            self.max_index[i,] = max_index 
            yi = flat_x[np.arange(flat_x.shape[0]), max_index].reshape(-1, self.input_shape[0]).T.ravel()
            y[i,] = yi.ravel()
        return y

    def backward(self, dy):
        N = dy.shape[0]
        dx = np.zeros([N, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        for i in range(N):
            dyi = dy[i,].reshape(self.shape).transpose([1, 2, 0]).ravel()
            dflat_x = np.zeros([np.prod(self.shape, dtype=int), np.prod(self.kernel_shape, dtype=int)])
            row = np.arange(np.prod(self.shape, dtype=int))
            dflat_x[np.arange(np.prod(self.shape, dtype=int)), self.max_index[i,]] = dyi
            dxi = utils.unflatten(dflat_x, 
                                 self.input_shape, 
                                 self.kernel_shape, 
                                 self.pad, 
                                 self.stride, 
                                 self.indice)
            dx[i, :] = dxi
        return dx
    
    def learn(self, config):
        pass