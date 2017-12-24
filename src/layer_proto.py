import numpy as np
import utils as utils 

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


class Conv(Layer):
    def __init__(self, kernel_h, kernel_w, output_d, pad, stride):
        super(ConvLayer, self).__init__()
        self.type = 'Conv'
        self.k_h = kernel_h
        self.k_w = kernel_w
        self.d = output_d
        self.pad = pad
        self.stride = stride

    def accept(self, src):
        src_d, src_h, src_w = src[0], src[1], src[2]
        if (src_h + 2 * self.pad - self.kernel_h + 1) % self.stride != 0:
            return False
        
        if (src_w + 2 * self.pad - self.kernel_w + 1) % self.stride != 0:
            return False

        self.src_d = src_d
        self.src_h = src_h
        self.src_w = src_w
        self.src = [self.src_d, self.src_h, src.src_w]

        self.k_d = src_d
        self.k = [self.k_d, self.k_h, self.k_w]
        self.ksize = np.prod(self.k, dtype=int)

        self.h_pos = utils.get_pos(self.src_h, self.kernel_h, self.pad, self.stride)
        self.w_pos = utils.get_pos(self.src_w, self.kernel_w, self.pad, self.stride)
        self.h = self.h_pos.size
        self.w = self.w_pos.size
        self.shape = [self.d, self.h, self.w]
        
        self.w = np.random.normal(0, 1, [self.ksize, self.d])
        self.b = np.random.normal(0, 1, [1, self.d])
        self.xcol = np.zeros([self.h * self.w, self.ksize])
        return True 

    def feed_forward(self, x):
        N = x.shape[0]
        self.xcol = utils.im2col(x)
        ycol = self.xcol @ self.w + self.b
        return ycol.reshape([N, -1], )

    def feed_backward(self, dy):
        dyConv = dy
        dxcol, dw, db = utils.gradient(dy, self.xcol, self.w)
        dx = utils.col2im(dxcol)
        self.w -= dw
        self.b -= db
        return dx

class Input(layer):
    def accept(self, src):
        pass

class FullyConnected(Layer):
    def __init__(self, output):
        super(FullyConncected, self).__init__()
        self.output = output
    
    def accept(self, src):
        self.input = np.prod(src, dytype=int)
        self.w = np.random.normal(0, 1, [self.input, self.output])
        self.b = np.random.normal(0, 1, [1, self.output])
        self.shape = [self.output]
        self.x = None
        
    def feed_forward(self, x):
        y = utils.predict(x, self.w, self.b) 
        self.x = x
        return y

    def feed_backward(self, dy):
        dx, dw, db = utils.gradient(dy, self.x, self.w)
        self.w -= dw
        self.b -= db
        return dx

class MaxPooling(Laye):
    def __init__(self, kernel_h, kernel_w):
    def accept(self, src):

        pass

class Relu(Layer):
    def accpet(self, src):
        pass
