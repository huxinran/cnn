"""
fully connected layer class
"""
import numpy as np
class FullyConnectedLayer:
    '''
    Fully Connected Layer Class represents a general function f(x, w) = y
    it provides 3 utility functions


    List of Variable
    ============================================================================
      Name | Type             | Explanation                                    
    ============================================================================
      n    | int              | dimension of input                             
      m    | int              | dimension of output                            
      T    | int              | num of inputs
    ---------------------------------------------------------------------------
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
    @staticmethod
    def init_weight(n, m):
        '''
        init weight (with bias)
        '''
        sd = 1 / np.sqrt(n)
        return np.random.normal(0, sd, [n, m]), np.random.normal(0, sd, [1, m])

    @staticmethod
    def fwd(x, w, b):
        '''
        y = x * w + b
        '''
        return np.dot(x, w) + b

    @staticmethod
    def bwd(g_y, x, w):
        '''
        g_x = g_y * w
        g_w = x.T * g_y
        g_b = 1.T * g_y
        '''
        g_x = np.dot(g_y, w.T)
        g_w = np.dot(x.T, g_y)
        g_b = np.sum(g_y, axis = 0)
        return g_x, g_w, g_b


class Layer:
    def __init__(self, inputDim, size, stride, padded=0):
        pass

    def get_pos(n, f, p, s):
        return np.arange(-p, n + 1 + p - f, s)

    def get_conv_index(pixel_shape, filter_shape, p, s):
        pixel_index = np.arange(np.prod(pixel_shape).astype(int)).reshape(pixel_shape)
        row_pos = Layer.get_pos(pixel_shape[1], filter_shape[1], p, s)
        col_pos = Layer.get_pos(pixel_shape[2], filter_shape[2], p, s)
        conv_index = np.zeros([row_pos.size * col_pos.size, np.prod(filter_shape).astype(int)])
        r = 0
        for i in row_pos:
            for j in col_pos:
                conv_index[r, :] = pixel_index[:, i : i + filter_shape[1], j : j + filter_shape[2]].ravel()
                r += 1
        return conv_index
    

    def convert_pixel_to_conv(pixel, pixel_shape, filter_shape, p = 0, s = 1):
        conv_index = Layer.get_conv_index(pixel_shape, filter_shape, p, s)
        conv_pixel = np.array(conv_index)
        print(conv_pixel.shape)
        for i in range(conv_index.shape[0]):
            for j in range(conv_index.shape[1]):
                t = conv_index[i][j].astype(int)
                print(i, j , t)
                conv_pixel[i][j] = pixel[t] if t >= 0 and t < pixel.size else 0 
        return conv_pixel
    

