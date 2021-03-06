import unittest
import numpy as np

import sys

sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\')
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\layer\\')

from maxpool import MaxPoolLayer as MaxPool


class TestMaxPoolLayer(unittest.TestCase):
    def test_init(self):
        l = MaxPool((2, 2))
        pass
    
    def test_accept(self):
        l = MaxPool((2, 2))
        self.assertTrue(l.accept([1, 10, 10]))
        pass

    def test_forward(self):
        l = MaxPool()
        l.accept([2, 4, 4])
        x = np.arange(32).reshape(1, 2, 4, 4)
        y = l.forward(x)
        self.assertTrue(np.allclose(y, [5, 7, 13, 15, 21, 23, 29, 31]))

    def test_backward(self):
        l = MaxPool()
        l.accept([2, 4, 4])
        x = np.arange(32).reshape(1, 32)
        y = l.forward(x)
        
        dy = np.array([[[[1, 1], 
                        [1, 1]], 
                       [[2, 2], 
                        [2, 2]]]])
        dx = l.backward(dy)

        self.assertTrue(np.allclose(dx, [[0, 0, 0, 0,
                                          0, 1, 0, 1,
                                          0, 0, 0, 0,
                                          0, 1, 0, 1, 
                                          0, 0, 0, 0,
                                          0, 2, 0, 2,
                                          0, 0, 0, 0,
                                          0, 2, 0, 2]]))

    def test_repr(self):
        l = MaxPool()
    
if __name__ == "__main__":
    unittest.main()