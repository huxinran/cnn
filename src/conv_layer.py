import numpy as np

class ConvLayer:
    def __init__(self, ishape, fshape, do, padding=0, stride=1):        
        self.di, self.hi, self.wi = ishape[0], ishape[1], ishape[2] 
        self.df, self.hf, self.wf = ishape[0], fshape[0], fshape[1]        
        self.i = ConvLayer.get_pos(self.ih, self.fh, self.padding, self.stride)
        self.j = ConvLayer.get_pos(self.iw, self.fw, self.padding, self.stride)
        self.do, self.ho, self.wo = do, self.i.size, self.j.size
        self.p = padding
        self.s = stride
        self.w, self.b = self.init_weight()
        self.conv_index = ConvLayer.init_conv_index(self.ishape, self.fshape, self.padding, self.stride)

    def init_weight(self):
        w = np.random.normal(0, 1, [self.id * self.fh * self.fw, self.od])
        b = np.random.normal(0, 1, [1, self.od])
        return w, b
    
    def feed_forward_single(self, x):
        return (ConvLayer.conv(x, self.conv_idx).dot(self.w) + self.b).T.ravel()

    def feed_backward_single(self, dy, convx, w):
        dy = dy.reshape(self.do, -1).T
        dconvx = np.dot(dy, w.T)
        dw = np.dot(convx.T, dy)
        db = np.sum(dy, axis = 0)
        dx = dconv()


    def feed_forward(self, x):
        N = x.shape[0]
        y = np.zeros([N, np.prod(self.oshape).astype(int)])
        for i in range(N):
            y[i, :] = get_y(x[i, :])
        return y

    def feed_backward(self, d_y):
        
        pass


    @staticmethod
    def get_pos(n, f, p, s):
        return np.arange(-p, n + 1 + p - f, s)

    @staticmethod
    def flat2conv(ishape, fshape, p, s):        
        _, ih, iw = ishape
        _, fh, fw = fshape

        row = ConvLayer.get_pos(ih, fh, p, s)
        col = ConvLayer.get_pos(iw, fw, p, s)
        conv_index = np.zeros([row.size * col.size, np.prod(fshape)])
        
        index = np.arange(np.prod(ishape)).reshape(ishape)
        r = 0
        for i in row:
            for j in col:
                conv_index[r, :] = index[:, i:i+fh, j:j+fw].ravel()
                r += 1
        return conv_index
    
    def conv2flat()
    @staticmethod
    def conv(x, conv_index):
        conv_x = np.array(conv_index)
        for i in range(conv_index.shape[0]):
            for j in range(conv_index.shape[1]):
                t = conv_index[i][j].astype(int)
                conv_x[i][j] = x[t] if t >= 0 and t < x.size else 0 
        return conv_x
    