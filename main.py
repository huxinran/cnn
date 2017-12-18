import numpy as np
import matplotlib.pyplot as plt 
plt.ion()

from network import NeuralNet
from utils import dataGen
from utils import softmax
from utils import mnist

from time import time

def main():

    (data, label) = mnist()
    data = np.array(data)
    label = np.array(label)
    din = data.shape[1]
    dhidden = 100
    dout = 10
    
    n = NeuralNet([din, dhidden, dout])   

    stepSize = 0.1
    iteration = 1000
    regularization = 0.0
    debug = False

    n.train(data, label, iteration, stepSize, regularization, debug=debug, testPct=0.001)
    n.show()
    return

if __name__ == "__main__":
    main()