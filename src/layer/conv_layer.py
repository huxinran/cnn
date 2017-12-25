from layer import Layer 
import utils
import numpy as np



class ConvLayer(Layer):
    def __init__(self, kernel_shape, depth_out, pad = 0, stride = 1):
        
        super(ConvLayer, self).__init__()
        self.type = 'ConvLayer'
        self.height_k, self.width_k = kernel_shape
        self.depth_out = depth_out
        self.pad = pad
        self.stride = stride

    def accept(self, src):
        depth_in, height_in, width_in = src
        if (height_in + 2 * self.pad - self.height_k + 1) % self.stride != 0:
            return False
        
        if (width_in + 2 * self.pad - self.width_k + 1) % self.stride != 0:
            return False

        self.src = np.array([depth_in, height_in, width_in])
        self.depth_in = depth_in
        self.height_in = height_in
        self.width_in = width_in

        self.depth_k = depth_in
        
        self.kernel_len = self.depth_k * self.height_k * self.width_k

        self.height_pos = utils.get_pos(self.height_in, self.height_k, self.pad, self.stride)
        self.width_pos = utils.get_pos(self.width_in, self.width_k, self.pad, self.stride)
        self.height_out = self.height_pos.size
        self.width_out = self.width_pos.size
        
        self.shape = np.array([self.depth_out, self.height_out, self.width_out])
        
        self.w = np.random.normal(0, 1, [self.kernel_len, self.depth_out])
        self.b = np.random.normal(0, 1, [1, self.depth_out])
        self.flat_x = None
        return True 

    def forward(self, x):
        '''
        x is 4-d 
        path 3 -d
        xpath is 2 -d
        ycol is 2 -d

        first split to make it 3 -d

        than transpose and ravel to make it 2-d

        '''
        N = x.shape[0]

        x = x.reshape(N, self.depth_in, self.height_in, self.width_in)
        
        y = np.zeros([N, self.depth_out * self.height_out * self.width_out]) 
        
        self.flat_x = np.zeros([N, self.height_out * self.width_out, self.height_k * self.width_k])

        for i in range(N):
            xi = x[i, :].reshape(self.depth_in, self.height_in, self.width_in)
            self.flat_x[i, :, :] = utils.flatten(xi, 
                                    [self.depth_in, self.height_in, self.width_in],
                                    [self.height_k, self.width_k], 
                                    self.pad, 
                                    self.stride) 

            y[i, :] = utils.forward(self.flat_x[i, :, :], self.w, self.b).T.ravel()
        
        return y

    def backward(self, dy):
        N = dy.shape[0]

        dx = np.zeros([N, self.depth_in * self.height_in * self.width_in])
        dw = np.zeros([self.height_k * self.width_k, self.depth_out])

        db = np.zeros([1, self.depth_out])
        for i in range(N):
            dyi = dy[i, :].reshape(self.depth_out, -1).T

            dflat_xi, dwi, dbi = utils.backward(dyi, self.flat_x[i, ], self.w)

            dxi = utils.unflatten(dflat_xi, 
                                        [self.depth_in, self.height_in, self.width_in],
                                        [self.height_k, self.width_k], 
                                        self.pad, 
                                        self.stride) 
            dx[i,] = dxi.ravel()
            dw += dwi
            db += dbi
        
        self.w -= dw
        self.b -= db

        return dx