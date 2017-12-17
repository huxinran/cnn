import numpy as np
import matplotlib.pyplot as plt 
from layer import FullyConnectedLayer 
from network import NeuralNet
from utils import dataGen
from utils import plot
from utils import softmax
import time


def main():
    N = 200
    din = 2
    dhidden = [200, 200]
    dout = 4
    n = NeuralNet(din, dout, dhidden)    
   
    trainData, trainLabel = dataGen(N, din)
    testData, testLabel = dataGen(N, din)

    r = 0.01
    T = 150

    # magic happens here
    n.train(trainData, trainLabel, r, T, testData=testData, testLabel=testLabel)
    
    return

if __name__ == "__main__":
    main()