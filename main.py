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

    r = 0.01
    T = 100

    # magic happens here
    n.train(data, label, r, T, testData=testData, testLabel=testLabel)
    #n.show()
    return

if __name__ == "__main__":
    main()