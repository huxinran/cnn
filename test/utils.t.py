import unittest
import numpy as np

import sys
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\')
import utils

class TestUtils(unittest.TestCase):
    def test_get_pos(self):
        self.assertTrue(np.array_equal(utils.get_pos(3, 2, 0, 1), [0, 1]))
        self.assertTrue(np.array_equal(utils.get_pos(7, 3, 0, 1), [0, 1, 2, 3, 4]))
        self.assertTrue(np.array_equal(utils.get_pos(5, 3, 1, 2), [0, 2, 4]))


    def test_softmax(self):
        self.assertTrue(np.array_equal(utils.softmax([[0, 0]]), [[0.5, 0.5]]))
        self.assertTrue(np.array_equal(utils.softmax([[10, 10, 10]]), [[1/3, 1/3, 1/3]]))
        self.assertTrue(np.allclose(utils.softmax([[10.0, 10.0], [0.0,  1.0]]),   
                                                  [[0.5, 0.5], [0.26894142, 0.73105858]]))

    def test_cross_entropy(self):
        self.assertTrue(np.array_equal()



if __name__ == "__main__":
    unittest.main()