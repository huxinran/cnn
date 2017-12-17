import numpy as np
import matplotlib.pyplot as plt 
from layer import FullyConnectedLayer 
from network import NeuralNet
from utils import dataGen
from utils import plot
from utils import softmax



def main():
    N = 100
    din = 2
    dhidden = [100, 100]
    dout = 5
    n = NeuralNet(din, dout, dhidden)    
   
    data, label = dataGen(N, din)
    testData, testLabel = dataGen(N, din)

    stepSize = 0.01
    iteration = 100
    regularization = 0.0001
    # magic happens here
    n.train(data, label, stepSize, iteration, regularization)
    #n.show()
    return

if __name__ == "__main__":
    main()