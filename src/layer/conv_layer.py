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
        self.patches = None
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
        x = x.reshape(-1, self.depth_in, self.height_in, self.width_in)

        N, depth_in, height_in, width_in = x.shape

        patch = utils.flatten(x, [self.height_k, self.width_k], self.pad, self.stride)        
        
        self.xpatch = patch.reshape(N * self.height_out * self.width_out, -1)
                
        ycol = utils.forward(self.xpatch, self.w, self.b)

        y = np.array(np.split(ycol, N)).transpose(0, 2, 1).reshape(N, -1)
        
        return y

    def backward(self, dy):
        N = dy.shape[0]

        dy = dy.reshape(N, self.depth_out, -1).transpose(0, 2, 1).reshape(-1, self.depth_out)
        
        dpatch, dw, db = utils.backward(dy, self.xcol, self.w)

        dpatch = dpatch.reshape([N, self.height_out * self.width_out, -1])


        dx = utils.unflatten(dpatch, 
                             [N, self.depth_in, self.height_in, self.width_in],
                             [self.height_k, self.width_k],
                             self.pad, 
                             self.stride)
        return dx, dw, db