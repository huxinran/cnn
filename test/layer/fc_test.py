import unittest
import numpy as np

import sys
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\')
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\layer\\')

from fc import FullyConnectedLayer

class TestFCLayer(unittest.TestCase):
    def test_init(self):
        config = {
            'shape' : 10
        }
        l = FullyConnectedLayer(config)
        pass
    
    def test_accept(self):
        config = {
            'shape' : 10
        }
        l = FullyConnectedLayer(config)
        self.assertTrue(l.accept(100))
        pass

    def test_forward(self):
        config = {
            'shape' : 2
        }
        l = FullyConnectedLayer(config)
        l.accept(3)

        x = np.array([[1, 2, 3], [4, 5, 6]])
        l.param['w'] = np.array([[1, 0], [0, 1], [1, 1]])
        l.param['b'] = np.array([[1, -2]])
        y, cache = l.forward(x)

        self.assertTrue(np.allclose(cache['x'], x))
        self.assertTrue(np.allclose(y, [[5, 3], [11, 9]]))


    def test_backward(self):
        config = {
            'shape' : 2
        }
        l = FullyConnectedLayer(config)
        l.accept(3)

        l.cache['x'] = np.array([[1, 2, 3], [4, 5, 6]])
        l.param['w'] = np.array([[1, 0], [0, 1], [1, 1]])
        l.param['b'] = np.array([[1, -2]])
        
        dy = np.array([[1, 1], [-1, 2]])
        dx, dparam = l.backward(dy)
        self.assertTrue(np.allclose(dx, [[1, 1, 2], [-1, 2, 1]]))        
        self.assertTrue(np.allclose(dparam['w'], [[-3, 9], [-3, 12], [-3, 15]]))
        self.assertTrue(np.allclose(dparam['b'], [0, 3]))

    def test_repr(self):
        config = {
            'shape' : 2
        }
        l = FullyConnectedLayer(config)
        l.accept(3)
        pass
    


if __name__ == "__main__":
    unittest.main()