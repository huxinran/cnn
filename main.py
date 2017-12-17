import numpy as np
import matplotlib.pyplot as plt 

plt.ion()
from layer import FullyConnectedLayer as FC
from network import NeuralNet
from utils import dataGen
from utils import plot
from utils import softmax
from time import time
from utils import cifar
from utils import mnist

def main():
    N = 200
 
    #data, label = dataGen(N, din)


    (data, label) = mnist()


    data = np.array(data)
    label = np.array(label)

    data = data - np.mean(data, axis=0)


    din = data.shape[1]
    dh1 = 100
    dh2 = 100
    dout = 10
    seed = 42
    np.random.seed(seed)
    
    n = NeuralNet([din, 128, dout])   

    stepSize = 1.0
    iteration = 1000
    regularization = 0.0
    debug = False

    n.train(data, label, iteration, stepSize, regularization, debug=debug, testPct=0.001)
    n.show()
    return

if __name__ == "__main__":
    main()