import unittest
import numpy as np

import sys
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\')
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\layer\\')
from lstm import LSTMLayer as LSTM
import utils

class TestLSTMLayer(unittest.TestCase):
    def test_init(self):
        config = {
            'dim_hidden' : 10
          , 'l' : 10
          , 'clip' : 5
        }
        l = LSTM(config)
        pass
    
    def test_accept(self):
        config = {
            'dim_hidden' : 10
          , 'l' : 10
          , 'clip' : 5
        }
        l = LSTM(config)
        l.accept([27])
        pass

    def test_forward(self):
        config = {
            'dim_hidden' : 10
          , 'l' : 10
          , 'clip' : 5
        }
        l = LSTM(config)
        l.accept([2])

        x = [np.array([0, 1])]
        y = l.forward(x)
        pass

    def test_backward(self):
        config = {
            'dim_hidden' : 3
          , 'l' : 10
          , 'clip' : 5
        }
        l = LSTM(config)
        l.accept([2])
        
        x = [np.array([[0, 1]])]
        y = l.forward(x)
        dy = [np.array([[0, 1]])]
        d = l.backward(dy, np.array([0.1]), np.array([0.1]))
        print(d)      
        pass
    
    def test_fit(self):
        pass

    def test_repr(self):
        pass

if __name__ == "__main__":
    unittest.main()