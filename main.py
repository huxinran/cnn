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

def main():
    """
    main func
    """
    
    (data, label) = mnist()
    data = np.array(data)
    label = np.array(label)
    din = data[0].size
    dhidden = 100
    dout = np.unique(label).size
    step_size = 1
    iteration = 1000
    regularization = 0.0001
    debug = False
    np.random.seed(42)

    n = Net([1, 28, 28])
    c = Conv([3, 3], 20)
    fc1 = FC([128])
    fc2 = FC([10])


   
    n.add_layer(c)
    n.add_layer(fc1)
    n.add_layer(fc2)
    
    print(n)
    n.fit(data, label, 10)


if __name__ == "__main__":
    main()
