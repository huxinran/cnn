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

def flatten_index(img_shape, kernel_shape, pad, stride):
    '''
    used to vectorize flatten
    '''
    #assert(img_shape.ndim == 3)
    #assert(kernel_shape.ndim == 2)
    
    depth_img, height_img, width_img = img_shape
    height_k, width_k = kernel_shape    
    

    k, i, j = np.meshgrid(np.arange(depth_img), 
                          np.arange(height_k), 
                          np.arange(width_k), 
                          indexing='ij')

    height_pad, width_pad = pad
    height_stride, width_stride = stride
    i_pos = get_pos(height_img, height_k, height_pad, height_stride)
    j_pos = get_pos(width_img, width_k, width_pad, width_stride)
    
    i_base, j_base = np.meshgrid(i_pos, j_pos, indexing='ij')
    
    i = np.tile(i.ravel(), i_pos.size * j_pos.size) + np.repeat(i_base.ravel(), depth_img * height_k * width_k)
    j = np.tile(j.ravel(), i_pos.size * j_pos.size) + np.repeat(j_base.ravel(), depth_img * height_k * width_k)
    k = np.tile(k.ravel(), i_pos.size * j_pos.size)
    return (k, i, j)

def flatten(img, img_shape, kernel_shape, pad, stride, indice=None):
    '''
    flatten a 3-d img into a 2-d array of patches
    ith row of patch is pixel of the ith location of patch arranged in [d, h, w] order  
    '''
    padded_img = pad_img(img, pad)
    
    if indice is None:
        k, i, j = flatten_index(img_shape, kernel_shape, pad, stride)
    else:
        k, i, j = indice 

    return padded_img[k, i, j]

def unflatten(patch, img_shape, kernel_shape, pad, stride, indice=None):
    '''
    unflatten 2-d array into a a 3-d img 
    ith row of col is pixel of the ith patch arranged by [d, h, w] order  
    '''
    
    padded_img = pad_img(np.zeros(img_shape), pad)

    if indice is None:
        k, i, j = flatten_index(img_shape, kernel_shape, pad, stride)
    else:
        k, i, j = indice 

    np.add.at(padded_img, (k, i, j), patch.ravel())

    return unpad_img(padded_img, pad)

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
