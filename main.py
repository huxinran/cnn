"""
main
"""
import numpy as np
import matplotlib.pyplot as plt
from src.net import Net
from data.data import mnist
from data.data import cifar
from data.data import getty
import sys
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\')
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\layer')

plt.ion()
from fc import FullyConnectedLayer as FC
from conv import ConvLayer as Conv
from relu import ReluLayer as Relu
from maxpool import MaxPoolLayer as MaxPool
from rnn import RNNLayer as RNN
from lstm import LSTMLayer as LSTM

from utils import normalize
import keras

def main():
    """
    main func
    """
<<<<<<< HEAD
    
    text, x, y, char2idx, idx2char = getty()
    T = 100
       
    config = {
        'dim_hidden' : 200
      , 'l' : T
      , 'clip' : 5
      , 'mu' : 0.9
      , 'step_size' : 0.01
    }
=======
>>>>>>> f5c76a2e0732c4a18754ebb8a30db5b5142511d3

    print(1)



if __name__ == "__main__":
    main()
