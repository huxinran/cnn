import unittest
import numpy as np

import sys
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\layer\\')

from fc import FullyConnectedLayer

class TestFCLayer(unittest.TestCase):
    def test_init(self):
        l = FullyConnectedLayer(10)
        pass
    
    def test_accept(self):
        l = FullyConnectedLayer(10)
        self.assertTrue(l.accept(100))
        pass

    def test_forward(self):
        l = FullyConnectedLayer(2)
        l.accept(3)

        x = np.array([[1, 2, 3], [4, 5, 6]])
        l.w = np.array([[1, 0], [0, 1], [1, 1]])
        l.b = np.array([[1, -2]])
        y = l.forward(x)

        self.assertTrue(np.allclose(y, [[5, 3], [11, 9]]))


    def test_backward(self):
        l = FullyConnectedLayer(2)
        l.accept(3)

        l.x = np.array([[1, 2, 3], [4, 5, 6]])
        l.w = np.array([[1, 0], [0, 1], [1, 1]])
        l.b = np.array([[1, -2]])
        dy = np.array([[1, 1], [-1, 2]])
        dx, dw, db = l.backward(dy)

        self.assertTrue(np.allclose(dx, [[1, 1, 2], [-1, 2, 1]]))        
        self.assertTrue(np.allclose(dw, [[-3, 9], [-3, 12], [-3, 15]]))
        self.assertTrue(np.allclose(db, [0, 3]))

    def test_repr(self):
        l = FullyConnectedLayer([10])
        pass
    


if __name__ == "__main__":
    unittest.main()