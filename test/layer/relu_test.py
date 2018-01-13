import unittest
import numpy as np

import sys
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\layer\\')

from relu import ReluLayer as Relu


class TestReluLayer(unittest.TestCase):
    def test_init(self):
        config = {}
        l = Relu(config)
        pass
    
    def test_accept(self):
        config = {}
        l = Relu(config)
        self.assertTrue(l.accept(100))
        pass

    def test_forward(self):
        config = {}
        l = Relu(config)
        l.accept([2, 3])

        x = np.array([[1, 2, 3, 
                      -4, -5, -6]])
        y, cache = l.forward(x)
        self.assertTrue(np.allclose(y, [[1, 2, 3, 0, 0, 0]]))
        self.assertTrue(np.allclose(cache['mask'], [[1, 1, 1, 0, 0, 0]]))


    def test_backward(self):
        config = {}
        l = Relu(config)
        l.accept([2, 2])
        l.cache['mask'] = np.array([[1, 0], [0, 1]])

        dy = np.array([[1, 1], [-1, 2]])
        
        dx, dparam = l.backward(dy)

        self.assertTrue(np.allclose(dx, [[1, 0], [0, 2]]))

    def test_repr(self):
        config = {}
        l = Relu(config)
        pass
    


if __name__ == "__main__":
    unittest.main()