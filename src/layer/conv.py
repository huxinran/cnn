from layer import Layer 
import utils
import numpy as np

class ConvLayer(Layer):
    def __init__(self, kernel_shape, depth_out, pad=0, stride=1):        
        super(ConvLayer, self).__init__()
        self.type = 'Convolution'
        self.kernel_shape = kernel_shape
        self.height_k, self.width_k = kernel_shape
        self.depth_out = depth_out
        self.pad = pad
        self.stride = stride

    def accept(self, input_shape):
        depth_in, height_in, width_in = input_shape[0], input_shape[1], input_shape[2]
        
        if (height_in + 2 * self.pad - self.height_k + 1) % self.stride != 0:
            return False
        
        if (width_in + 2 * self.pad - self.width_k + 1) % self.stride != 0:
            return False

        self.input_shape = np.array([depth_in, height_in, width_in])        
        self.d_in = np.prod(self.input_shape, dtype=int)

        self.depth_in, self.height_in, self.width_in = depth_in, height_in, width_in
        
        self.depth_k = depth_in

        self.height_pos = utils.get_pos(self.height_in, self.height_k, self.pad, self.stride)
        self.width_pos = utils.get_pos(self.width_in, self.width_k, self.pad, self.stride)
        
        self.height_out = self.height_pos.size
        self.width_out = self.width_pos.size
        
        self.shape = np.array([self.depth_out, self.height_out, self.width_out])
        
        
        self.indice = utils.flatten_index(self.input_shape, self.kernel_shape, 
                                          (self.pad, self.pad),
                                          (self.stride, self.stride)) 

        # params
        kernel_len = self.depth_k * self.height_k * self.width_k
        self.w = np.random.normal(0, 1, [kernel_len, self.depth_out])
        self.b = np.random.normal(0, 1, [1, self.depth_out])
        
        # cache
        self.flat_x = None
        self.dw = None
        self.db = None
        
        return True 

    def forward(self, x):
        '''
        '''
        N = x.shape[0]

        x = x.reshape(N, self.depth_in, self.height_in, self.width_in)
        
        y = np.zeros([N, np.prod(self.shape, dtype=int)]) 
        
        self.flat_x = np.zeros((N, self.height_out * self.width_out, self.height_k * self.width_k))

        for i in range(N):
            xi = x[i, :].reshape(self.depth_in, self.height_in, self.width_in)
            self.flat_x[i, :, :] = utils.flatten(xi, 
                                                 self.input_shape, 
                                                 self.kernel_shape, 
                                                 (self.pad, self.pad),
                                                 (self.stride, self.stride), 
                                                 self.indice).reshape(self.height_out * self.width_out, -1) 

            y[i, :] = utils.forward(self.flat_x[i, :, :], self.w, self.b).T.ravel()
        
        return y

    def backward(self, dy):
        N = dy.shape[0]

        dx = np.zeros([N, self.d_in], dtype=float)
        dw = np.zeros([self.height_k * self.width_k, self.depth_out], dtype=float)
        db = np.zeros([1, self.depth_out], dtype=float)

        for i in range(N):
            dyi = dy[i, :].reshape(self.depth_out, -1).T

            dflat_xi, dwi, dbi = utils.backward(dyi, self.flat_x[i, ], self.w)

            dxi = utils.unflatten(dflat_xi, 
                                  self.input_shape,
                                  self.kernel_shape,
                                  (self.pad, self.pad),
                                  (self.stride, self.stride),
                                  self.indice) 
            dx[i,] = dxi.ravel()
            dw += dwi
            db += dbi

        self.dw = dw
        self.db = db
        return dx
    
    def learn(self, config):
        self.w -= config['step_size'] * self.dw
        self.b -= config['step_size'] * self.db