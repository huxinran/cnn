"""
main
"""
import numpy as np
import matplotlib.pyplot as plt
from src.net import Net
from data.data import mnist
import sys
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\')
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\layer')

plt.ion()
from fc import FullyConnectedLayer as FC
from conv import ConvLayer as Conv
from relu import ReluLayer as Relu
from maxpool import MaxPoolLayer as MaxPool


def main():
    """
    main func
    """
    
    (data, label) = mnist()
    data = np.array(data)[0:1000, ]
    label = np.array(label)[0:1000, ]
    din = data[0].size
    dhidden = 100
    dout = np.unique(label).size
    
    step_size = 1
    iteration = 1000
    regularization = 0.0001
    debug = False
    np.random.seed(42)

    n = Net([1, 28, 28])
    conv = Conv([3, 3], 6)
    relu = Relu()
    pool = MaxPool()
    fc = FC([10])

    n.add(conv)
    n.add(relu)
    n.add(pool)
    n.add(fc)
    
    print(n)
    n.fit(data, label, 20)


if __name__ == "__main__":
    main()
