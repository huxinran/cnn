import numpy as np
import matplotlib.pyplot as plt 
from layer import FullyConnectedLayer as FC
from network import NeuralNet
from utils import dataGen
from utils import plot
from utils import softmax
from time import time


def main():
    N = 200
    din = 2
    dh1 = 100
    dh2 = 100
    dout = 4
    seed = 42
    np.random.seed(seed)
    
    n = NeuralNet([din, dh1, dh2, dout])    
    data, label = dataGen(N, din)
    stepSize = 0.01
    iteration = 1000
    regularization = 0.0
    debug = False

    n.train(data, label, iteration, stepSize, regularization, debug=debug, testPct=0.2)
    n.show()
    return

if __name__ == "__main__":
    main()