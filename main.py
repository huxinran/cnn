import numpy as np
import matplotlib.pyplot as plt 
plt.ion()

from network import NeuralNet
from utils import dataGen
from utils import softmax
from utils import mnist

from time import time

def main():

    (data, label) = dataGen(200, 2)
    data = np.array(data)
    label = np.array(label)
    din = data[0].size
    dhidden = 128
    dout = np.unique(label).size
    stepSize = 1
    iteration = 1000
    regularization = 0.0
    debug = False
    
    net = NeuralNet([din, dhidden, dout])   
    
    # magic happens
    net.train(data, label, iteration, stepSize, regularization, debug=debug, testPct=0.001)
    net.show()
    return

if __name__ == "__main__":
    main()