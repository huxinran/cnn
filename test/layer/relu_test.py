import unittest
import numpy as np

import sys
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\layer\\')

from relu import ReluLayer as Relu


class TestReluLayer(unittest.TestCase):
    def test_init(self):
        l = Relu()
        pass
    
    def test_accept(self):
        l = Relu()
        self.assertTrue(l.accept(100))
        pass

    def test_forward(self):
        l = Relu()
        l.accept([2, 3])

        x = np.array([[1, 2, 3, 
                      -4, -5, -6]])
        y = l.forward(x)
        self.assertTrue(np.allclose(y, [[1, 2, 3, 0, 0, 0]]))

    def test_backward(self):
        l = Relu()
        l.accept([2, 2])
        l.x = np.array([[1, -1], [-1, 1]])

        dy = np.array([[1, 1], [-1, 2]])
        
        dx = l.backward(dy)

        self.assertTrue(np.allclose(dx, [[1, 0], [0, 2]]))

    def test_repr(self):
        l = Relu()
        pass
    


if __name__ == "__main__":
    unittest.main()