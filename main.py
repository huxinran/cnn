"""
main
"""
import numpy as np
import matplotlib.pyplot as plt
from src.net import Net
from data.data import mnist
from data.data import cifar

import sys
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\')
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\layer')

plt.ion()
from fc import FullyConnectedLayer as FC
from conv import ConvLayer as Conv
from relu import ReluLayer as Relu
from maxpool import MaxPoolLayer as MaxPool
from utils import normalize

def main():
    """
    main func
    """
    
    (data, label) = cifar()
    print(data.shape)

    data = np.array(data, dtype=float)
    label = np.array(label)
    
    data = normalize(data)
    print(data.shape)
    din = data[0].size
    dhidden = 100
    dout = np.unique(label).size
    
    step_size = 1
    iteration = 1000
    regularization = 0.0001
    debug = False
    np.random.seed(42)

    n = Net([3, 32, 32])
    conv = Conv([3, 3], 20)
    relu = Relu()
    pool = MaxPool()
    fc = FC([10])

    n.add(conv)
    n.add(relu)
    n.add(pool)
    n.add(fc)
    
    print(n)
    n.fit(data, label, 200)


if __name__ == "__main__":
    main()
