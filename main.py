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
from fc_layer import FullyConnectedLayer as FC
from conv_layer import ConvLayer as Conv
from relu_layer import ReluLayer as Relu

def main():
    """
    main func
    """
    
    (data, label) = mnist()
    data = np.array(data)[0:100, ]
    label = np.array(label)[0:100, ]
    din = data[0].size
    dhidden = 100
    dout = np.unique(label).size
    step_size = 1
    iteration = 1000
    regularization = 0.0001
    debug = False
    np.random.seed(42)

    n = Net([1, 28, 28])
    conv = Conv([3, 3], 3)
    relu = Relu()
    fc = FC([10])

    n.add_layer(conv)
    n.add_layer(relu)
    n.add_layer(fc)
    
    print(n)
    n.fit(data, label, 20)


if __name__ == "__main__":
    main()
