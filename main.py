import numpy as np
import matplotlib.pyplot as plt 
plt.ion()

from network import NeuralNet
from utils import data_gen
from utils import softmax
from utils import mnist

from time import time

def main():

    (data, label) = mnist()
    data = np.array(data)
    label = np.array(label)
    din = data[0].size
    dhidden = 100
    dout = np.unique(label).size
    step_size = 1
    iteration = 1000
    regularization = 0.0
    debug = False
    
    net = NeuralNet([din, dhidden, dout])   
    
    # magic happens
    net.fit(data, label, iteration, step_size, regularization, debug=debug, test_pct=0.001)
    
    return

if __name__ == "__main__":
    main()