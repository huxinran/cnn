import numpy as np
import utils 



class CNN:
    def __init__(self):
        self.c = ConvLayer([1, 28, 28], [3, 3], 100)
        self.fc = ConvLayer(self.c.output_shape(), [26, 26], 10)

    def train_iteration(self, data, label):
        y1 = self.c.feed_forward(data)


        y2 = self.fc.feed_forward(y1)
    

        loss, dy = utils.compute_loss(y2, label)

        dy2, dw2, db2 = self.fc.feed_backward(dy)

        dy1, dw1, db1 = self.c.feed_backward(dy2)

        return loss, dw1, db1, dw2, db2

    def fit(self, data, label, iter):
        for t in range(iter):
            loss, dw1, db1, dw2, db2 = self.train_iteration(data, label)

            self.c.w -= 0.01 * dw1
            self.c.b -= 0.01 * db1
            self.fc.w -= 0.01 * dw2
            self.fc.b -= 0.01 * db2
            #print(loss)
            print(t, np.mean(loss))





class ConvLayer:
    '''
    Conv Layer Class represents conv layer that apply filters on an input image 
    and produce an output image

    * each image is accepted as a row array that ranged by the order of 
    [depth, height, width]

    * output shape is determined based on input and filter shape, 

    * w is a [f, do] array

    * b is a [1, do] array

    * one filter for each depth of output



    forward step
    (1) input x                     [1, di * hi * wi]
    (2) x => xconv                  [ho * wo, f] array
    (3) yconv = xconv * w + b       [ho * wo, do] array
    (4) yconv => y                  [1, do * ho * wo]
    
    backward
    (1) input dy                    [1, do * ho * wo]
    (2) dy => d_yconv               [ho * wo, do]
    (3) dconv, dw, db               [ho * wo, f], [f, d], [1, d] 
    (4) d_convx => dx               [1, d1 * h1 * wi]


    List of Variable
    ============================================================================
      Name       | Type             | Explanation                                    
    ============================================================================
      n          | int              | dimension of input                             
      m          | int              | dimension of output                            
      T          | int              | num of inputs
      di, hi, wi | shape of input
      do, ho, wo | shape of output
      df, hf, wf | shape of filter
      
    ----------------------------------------------------------------------------
      x    | (T, n)           | input                                 
      w    | (n, m)           | weight
      b    | (1, m)           | bias                        
      y    | (T, m)           | output                   
    ----------------------------------------------------------------------------
      g_y  | (T, m)           | gradient on output                             
      g_x  | (T, n)           | gradient on input                              
      g_w  | (n, m)           | gradient on weight
      g_b  | (1, m)           | gradient on bias                  
    ============================================================================
    '''

    def __init__(self, input_shape, filter_shape, output_depth, padding=0, stride=1):        
        
        self.di, self.hi, self.wi = input_shape[0], input_shape[1], input_shape[2]    
        self.df, self.hf, self.wf = input_shape[0], filter_shape[0], filter_shape[1]      
        
        self.row_pos = ConvLayer.get_pos(self.hi, self.hf, padding, stride)
        self.col_pos = ConvLayer.get_pos(self.wi, self.wf, padding, stride)

        self.do, self.ho, self.wo = output_depth, self.row_pos.size, self.col_pos.size
        self.p = padding
        self.s = stride
        self.w, self.b = self.init_weight()
        self.index = ConvLayer.init_conv_index(input_shape, filter_shape, padding, stride)
        self.xconv = None

        
    def init_weight(self):
        w = np.random.normal(0, 1, [self.df * self.hf * self.wf, self.do])
        b = np.random.normal(0, 1, [1, self.do])
        return w, b

    def output_shape(self):
        return [self.do, self.ho, self.wo]


    def feed_forward(self, x):
        N = x.shape[0]
        y = np.zeros([N, self.do * self.ho * self.wo])
        self.xconv = np.zeros([N, self.ho * self.wo, self.df * self.hf * self.wf])
        for i in range(N):
            y[i, :], self.xconv[i, :] = ConvLayer.fwd(x[i, :], self.w, self.b, self.index)
        return y

    def feed_backward(self, dy):
        N = dy.shape[0]
        dx = np.zeros([N, self.di * self.hi * self.wi])
        dw = np.zeros([self.df * self.hf * self.wf, self.do])
        db = np.zeros([1, self.do])
        for i in range(N):
            dxi, dwi, dbi = ConvLayer.bwd(dy[i,:], self.xconv[i, :, :], self.w, self.index)
            dx[i, :] = dxi
            dw += dwi
            db += dbi
        return dx, dw / N, db / N


    @staticmethod
    def get_pos(n, f, p, s):
        return np.arange(-p, n + 1 + p - f, s)

    @staticmethod
    def init_conv_index(ishape, fshape, p=0, s=1):        
        di, hi, wi = ishape
        hf, wf = fshape
        l = np.prod(ishape).astype(int)

        row = ConvLayer.get_pos(hi, hf, p, s)
        col = ConvLayer.get_pos(wi, wf, p, s)
        conv_index = np.zeros([row.size * col.size, di * hf * wf], dtype=int)
        flat_index = np.arange(np.prod(ishape)).reshape(ishape)

        r = 0
        for i in row:
            for j in col:
                conv_index[r, :] = flat_index[:, i:i+hf, j:j+wf].ravel().astype(int)
                r += 1
        return conv_index
    
    @staticmethod
    def flat2conv(flat, index):
        #print(np.amax(index))
        return flat[index.ravel()].reshape(index.shape)
        l = flat.size
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                t = index[i][j].astype(int)
                if t >= 0 and t < l:
                    conv[i][j] = flat[t]
        return conv
                
    @staticmethod
    def conv2flat(conv, index):
        
        flat = np.zeros(np.amax(index).astype(int) + 1)
        l = flat.size
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                t = index[i][j].astype(int)
                if t >= 0 and t < l:
                    flat[t] += conv[i][j]
        return flat

    @staticmethod
    def fwd(x, w, b, index):
        xconv = ConvLayer.flat2conv(x, index)
        yconv = xconv @ w + b
        return yconv.T.ravel(), xconv
    
    @staticmethod
    def bwd(dy, xconv, w, index):
        dyconv = dy.reshape(-1, xconv.shape[0]).T
        dxconv = dyconv @ w.T
        dw = xconv.T @ dyconv
        db = np.sum(dyconv, axis = 0)
        dx = ConvLayer.conv2flat(dxconv, index)
        return dx, dw, db
    