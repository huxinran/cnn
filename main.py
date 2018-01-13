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

def main():
    """
    main func
    """
    
    text, x, y, char2idx, idx2char = getty()
    T = 100
       
    config = {
        'dim_hidden' : 200
      , 'l' : T
      , 'clip' : 5
      , 'mu' : 0.9
      , 'step_size' : 0.01
    }

    #np.random.seed(42)
    r = RNN(config)
    r.accept([27])
    m = 1
    ttb = r.sample('f', T * m, char2idx, idx2char, **r.model)
    r.fit(x[:T * m], y[:T * m ], T, 100000, char2idx, idx2char)
    tta = r.sample('f', T * m, char2idx, idx2char, **r.model)
    print(ttb)
    print(tta)
    print(text[:T * m])
    return 




    (data, label) = cifar()
    N = 10000
    data = np.array(data, dtype=float)[:N,]
    label = np.array(label)[:N,]

    data = normalize(data)

    config = {
        'input_shape' : [3, 32, 32]
      , 'mu' : 0.9
      , 'step_size' : 0.000001
      , 'step_decay' : 0.95
    }

    nn = Net(config)
    conv1 = Conv([3, 3], 6)
    relu1 = Relu()
    conv2 = Conv([3, 3], 32)
    relu2 = Relu()
    pool = MaxPool()
    fc = FC([10])

    nn.add(conv1)
    nn.add(relu1)
    nn.add(pool)
    nn.add(fc)
    
    print(nn)
    nn.fit(data, label, 200)


if __name__ == "__main__":
    main()
