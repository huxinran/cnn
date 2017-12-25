import numpy as np

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

def cross_entropy(p, l):
    """
    given the output and true label,
    return softmax cross entropy loss and gradient on output
    """
    loss = -np.log(np.maximum(np.exp(-10), p[np.arange(l.size), l]))
    p[np.arange(l.size), l] -= 1
    return loss, p

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
    if pad[0] == 0 and pad[1] == 0:
        return img
    elif pad > 0:
        return np.pad(img, ((0, 0), (pad[0], pad[0]), (pad[1], pad[1])), 'constant')

def unpad_img(padded_img, pad):
    '''
    remove padded zeros
    '''
    if pad[0] == 0 and pad[1] == 0:
        return padded_img
    else:
        return img[:, pad[0]:-pad[0], pad[1]:-pad[1]]

def flatten_index(shape, kernel_shape, pad, stride):
    '''
    used to vectorize flatten
    '''
    d, h, w = shape
    hk, wk = kernel_shape    
    
    k, i, j = np.meshgrid(np.arange(d), np.arange(hk), np.arange(wk), indexing='ij')

    hp, wp = pad
    hs, ws = stride
    ipos = get_pos(h, hk, hp, hs)
    jpos = get_pos(w, wk, wp, ws)
    ib, jb = np.meshgrid(ipos, jpos, indexing='ij')
    
    a = ipos.size * jpos.size
    b = d * hk * wk
    i = np.tile(i.ravel(), a) + np.repeat(ib.ravel(), b)
    j = np.tile(j.ravel(), a) + np.repeat(jb.ravel(), b)
    k = np.tile(k.ravel(), a)
    return (k, i, j)

def flatten(img, shape, kernel_shape, pad, stride, indice=None):
    '''
    flatten a 3-d img into a 2-d array of patches
    ith row of patch is pixel of the ith location of patch arranged in [d, h, w] order  
    '''
    padded_img = pad_img(img, pad)
    
    if indice is None:
        k, i, j = flatten_index(shape, kernel_shape, pad, stride)
    else:
        k, i, j = indice 

    return padded_img[k, i, j]

def unflatten(patch, shape, kernel_shape, pad, stride, indice=None):
    '''
    unflatten 2-d array into a a 3-d img 
    ith row of col is pixel of the ith patch arranged by [d, h, w] order  
    '''
    padded_img = pad_img(np.zeros(shape), pad)

    if indice is None:
        k, i, j = flatten_index(shape, kernel_shape, pad, stride)
    else:
        k, i, j = indice 

    np.add.at(padded_img, (k, i, j), patch.ravel())
    return unpad_img(padded_img, pad)

def split(data, label, pct=0.0):
    """
    split data, label into train and test set
    """
    n = np.ceil(label.size * (1 - pct), dtype=int)
    data1, label1 = data[:n, :], label[:n, :]
    data2, label2 = data[n:, :], label[n:, :] 
    return (data1, label1), (data2, label2)

def normalize(data):
    """
    normalize data
    """
    data = data - np.mean(data, axis=0)
    return data
