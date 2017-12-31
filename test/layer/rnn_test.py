import unittest
import numpy as np

import sys
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\layer\\')
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\')
from rnn import RNNLayer as RNN
import utils

class TestRNNLayer(unittest.TestCase):
    def test_init(self):
        config = {
            'dim_hidden' : 10
          , 'len' : 2
        }
        l = RNN(config)
        pass
    
    def test_accept(self):
        config = {
            'dim_hidden' : 10
          , 'len' : 2
        }
        l = RNN(config)
        l.accept([26])
        pass

    def test_forward(self):
        config = {
            'dim_hidden' : 10
          , 'len' : 2
        }
        l = RNN(config)
        l.accept([26])
        x = [np.zeros([26])] * 2
        x[0][0] = 1.0
        x[1][1] = 1.0
         
        l.forward(x)
        
        pass

    def test_backward(self):
        config = {
            'dim_hidden' : 10
          , 'len' : 2
        }
        l = RNN(config)
        l.accept([26])
        x = [np.zeros([26])] * 2
        x[0][0] = 1.0
        x[1][1] = 1.0
         
        y = l.forward(x)


        dy = [None] * 2
        loss, dy[0] = utils.cross_entropy(utils.softmax(y[0]), np.array([0]))
        loss, dy[1] = utils.cross_entropy(utils.softmax(y[1]), np.array([1]))
        
        dW, dU, dV = l.backward(dy)

    
    def test_fit(self):
        config = {
            'dim_hidden' : 10
          , 'len' : 2
          , 'step_size' : 0.01
        }
        l = RNN(config)
        l.accept([26])
        x = [np.zeros([26])] * 2
        x[0][0] = 1.0
        x[1][1] = 1.0
        
        y = np.array([1, 2])
        l.fit(x, y, 100, config)
        
    def test_repr(self):
        pass

if __name__ == "__main__":
    unittest.main()