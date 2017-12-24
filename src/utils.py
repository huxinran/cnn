import pickle
import numpy as np
import matplotlib.pyplot as plt 

def softmax(y):
    '''
    comput softmax of a list of value
    '''
    p = np.exp(y - np.amax(y, axis=1, keepdims=True))
    p /= np.sum(p, axis=1, keepdims=True)
    return p

def get_pos(m, n, pad=0, stride=1):
    '''
    get a list of pos for filter to start
    '''
    assert(pad >= 0)
    assert(stride >= 1)
    assert(m > 0)
    assert(n > 0)
    assert((m + 2 * pad - n) % stride == 0)

    return np.arange(0, m + 1 + 2 * pad - n, stride)

def cross_entropy(p, label):
    """
    given the output and true label,
    return softmax cross entropy loss and gradient on output
    """
    LOSS_CAP = -10
    row_idx = np.arange(label.size)
    loss = -np.log(np.maximum(np.exp(LOSS_CAP), p[row_idx, label]))
    dy = p
    dy[np.arange(label.size), label] -= 1
    return loss, dy

def forward(x, w, b):
    '''
    y = x * w + b
    '''
    return x @ w + b

def backward(dy, x, w):
    '''
    dx = dy * w
    dw = x.T * dy
    db = 1.T * dy
    '''
    dx = dy @ w.T
    dw = x.T @ dy
    db = np.sum(dy, axis=0)
    return dx, dw, db

def pad_img(img, pad):
    '''
    pad img with zeros
    '''
    if pad == 0:
        return img
    elif pad > 0:
        return np.pad(img, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')

def unpad_img(padded_img, pad):
    '''
    remove padded zeros
    '''
    if pad == 0:
        return padded_img
    else:
        return img[:, :, pad:-pad, pad:-pad]

def flatten_index(input_shape, kernel_shape, pad=0, stride=1):
    '''
    used to vectorize flatten
    '''
    din, hin, win = input_shape
    hk, wk = kernel_shape    
    
    k, i, j = np.meshgrid(np.arange(din), 
                          np.arange(hk), 
                          np.arange(wk), 
                          indexing='ij')

    i_pos = get_pos(hin, hk, pad, stride)
    j_pos = get_pos(win, wk, pad, stride)
    
    i_base, j_base = np.meshgrid(i_pos, j_pos, indexing='ij')
    
    i = np.tile(i.ravel(), i_pos.size * j_pos.size) + np.repeat(i_base.ravel(), din * hk * wk)
    
    j = np.tile(j.ravel(), i_pos.size * j_pos.size) + np.repeat(j_base.ravel(), din * hk * wk)
    
    k = np.tile(k.ravel(), i_pos.size * j_pos.size)
    return (k, i, j)

def flatten(img, kernel_shape, pad, stride):
    '''
    this will flatten a 3-d img into a 2-d array of patches
    ith row of col is pixel of the ith patch arranged by [d, h, w] order  
    '''
    N, depth_in, height_in, width_in = img.shape
    
    height_k, width_k = kernel_shape

    padded_img = pad_img(img, pad)

    k, i, j = flatten_index([depth_in, height_in, width_in], 
                            [height_k, width_k], 
                            pad, 
                            stride)
    
    patch = padded_img[:, k, i, j]

    return patch.reshape([N, -1, depth_in * height_k * width_k])

def unflatten(patch, img_shape, kernel_shape, pad, stride):
    '''
    unflatten 2-d array into a a 3-d img 
    ith row of col is pixel of the ith patch arranged by [d, h, w] order  
    '''
    N, depth_img, height_img, width_img = img_shape

    patch = patch.reshape([N, -1])

    padded_img = pad_img(np.zeros(img_shape), pad)

    k, i, j = flatten_index(img_shape[1:], kernel_shape, pad, stride)
    
    np.add.at(padded_img, (slice(None), k, i, j), patch)

    img = unpad_img(padded_img, pad)

    return img


def split(data, label, pct=0.0):
    """
    split data, label into train and test set
    """
    N = np.ceil(label.size * (1 - pct), dtype=int)
    data1, label1 = data[:N, :], label[:N, :]
    data2, label2 = data[N:, :], label[N:, :] 
    return data1, label1, data2, label2

def normalize(data):
    """
    normalize data
    """
    data = data - np.mean(data, axis=0)
    return data










# these may be deprecated 

def init_conv_index(input_shape, filter_shape, padding=0, stride=1):
    '''

    '''       
    di, hi, wi = inpit_shape
    hf, wf = filter_shape
    row_pos = get_pos(hi, hf, p, s)
    col_pos = get_pos(wi, wf, p, s)

    conv_index = np.zeros([row_pos.size * col_pos.size, di * hf * wf], dtype=int)
    flat_index = np.arange(np.prod(input_shape, dtype=int)).reshape(input_shape)
    r = 0
    for i in row_pos:
        for j in col_pos:
            conv_index[r, :] = flat_index[:, i : i + hf, j : j + wf].ravel()
            r += 1
    return conv_index

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
            

def conv2flat(conv, index):
    
    flat = np.zeros(np.amax(index).astype(int) + 1)
    l = flat.size
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            t = index[i][j].astype(int)
            if t >= 0 and t < l:
                flat[t] += conv[i][j]
    return flat

def fwd(x, w, b, index):
    xconv = ConvLayer.flat2conv(x, index)
    yconv = xconv @ w + b
    return yconv.T.ravel(), xconv


def bwd(dy, xconv, w, index):
    dyconv = dy.reshape(-1, xconv.shape[0]).T
    dxconv = dyconv @ w.T
    dw = xconv.T @ dyconv
    db = np.sum(dyconv, axis = 0)
    dx = ConvLayer.conv2flat(dxconv, index)
    return dx, dw, db
    











