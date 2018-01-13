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

from utils import normalize

def main():
    """
    main func
    """
    np.random.seed(42)
    (data, label) = cifar()
    N = 1000
    data = np.array(data, dtype=float)[:N,]
    label = np.array(label)[:N,]

    data = normalize(data)
    config = {
        'input_shape' : [3, 32, 32]
      , 'mu' : 0.9
      , 'step_size' : 10.0
      , 'step_decay' : 0.95
    }

    nn = Net(config)
    conv1 = Conv({
        'kernel_shape' : (3, 3)
      , 'output_depth' : 32
      , 'pad' : [0, 0]
      , 'stride' : [1, 1]
    })
    
    relu1 = Relu({})
    
    conv2 = Conv({
        'kernel_shape' : (3, 3)
      , 'output_depth' : 32
      , 'pad' : [0, 0]
      , 'stride' : [1, 1]
    })

    relu2 = Relu({})

    pool = MaxPool({
        'kernel_shape' : (2, 2)
    })
    
    fc = FC({
        'shape' : 10
    })

    nn.add(conv1)
    nn.add(relu1)
    nn.add(pool)
    nn.add(fc)
    
    print(nn)
    nn.fit(data, label, 1000)



if __name__ == "__main__":
    main()
