import unittest
import numpy as np

import sys
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\layer\\')

from maxpool_layer import MaxPoolLayer as MaxPool


class TestMaxPoolLayer(unittest.TestCase):
    def test_init(self):
        l = MaxPool(2, 2)
        pass
    
    def test_accept(self):
        l = MaxPool()
        self.assertTrue(l.accept([1, 10, 10]))
        pass

    def test_forward(self):
        l = MaxPool()
        l.accept([1, 4, 4])
        x = np.arange(16).reshape(4, 4)
        y = l.forward(x)
        self.assertTrue(np.allclose(y, [[5, 7], [13, 15]]))

    def test_backward(self):
        l = MaxPool()
        l.accept([1, 4, 4])
        l.xidx = np.array([[1, -1], [-1, 1]])
        dy = np.array([[1, 1], [-1, 2]])
        
        dx = l.backward(dy)

        self.assertTrue(np.allclose(dx, [[1, 0], [0, 2]]))

    def test_repr(self):
        l = MaxPool()
        print(l)
        pass
    