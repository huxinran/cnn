import unittest
import numpy as np

import sys
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\')
import utils

class TestUtils(unittest.TestCase):
    def test_get_pos(self):
        val = utils.get_pos(3, 2, 0, 1)
        self.assertTrue(np.array_equal(val, [0, 1]))

        val = utils.get_pos(7, 3, 0, 1)
        self.assertTrue(np.array_equal(val, [0, 1, 2, 3, 4]))

        val = utils.get_pos(5, 3, 1, 2)
        self.assertTrue(np.array_equal(val, [0, 2, 4]))


    def test_softmax(self):
        val = utils.softmax([[0, 0]])
        self.assertTrue(np.array_equal(val, [[0.5, 0.5]]))
        
        val = utils.softmax([[10, 10, 10]])
        self.assertTrue(np.array_equal(val, [[1/3, 1/3, 1/3]]))
        
        val = utils.softmax([[10.0, 10.0], [0.0,  1.0]])
        self.assertTrue(np.allclose(val, [[0.5, 0.5], [0.26894142, 0.73105858]]))

    def test_cross_entropy(self):
        loss, dy = utils.cross_entropy(np.array([[0, 1]]), np.array([0]))
        self.assertTrue(np.allclose(loss, [[10]]))
        self.assertTrue(np.allclose(dy,[[-1, 1]]))

        
        loss, dy = utils.cross_entropy(np.array([[0, 1]]), np.array([1]))
        self.assertTrue(np.allclose(loss, [[0]]))
        self.assertTrue(np.allclose(dy, [[0, 0]]))


        loss, dy = utils.cross_entropy(np.array([[0.5, 0.5]]), np.array([1]))
        self.assertTrue(np.allclose(loss, [[-np.log(0.5)]]))
        self.assertTrue(np.allclose(dy, [[0.5, -0.5]]))

    def test_forward(self):
        x = np.array([[1, 2, 3], [4, 5, 6]])
        w = np.array([[1, 0, -1], [0, 1, 1], [1, -1, 0]])
        b = np.array([[1, 3, 2]])
        y = utils.forward(x, w, b)
        self.assertTrue(np.allclose(y, [[5, 2, 3], [11, 2, 3]]))

    def test_backward(self):
        dy = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        x = np.array([[1, 2, 3], [4, 5, 6]])
        w = np.array([[1, 0, 0, 0], [-1, 0, 1, 0], [0, 2, -3, 1]])

        dx, dw, db = utils.backward(dy, x, w)
        self.assertTrue(np.allclose(dx, 
                                    [[1, 2, -1], 
                                     [5, 2, -1]]))
        
        self.assertTrue(np.allclose(dw, 
                                    [[21, 26, 31, 36], 
                                     [27, 34, 41, 48], 
                                     [33, 42, 51, 60]]))
        
        self.assertTrue(np.allclose(db, 
                                    [[6, 8, 10, 12]]))


    def test_flatten_index(self):
        [k, i, j] = utils.flatten_index((2, 3, 3), (2, 2), (0, 0), (1, 1))

        self.assertTrue(np.allclose(k, [0, 0, 0, 0, 1, 1, 1, 1,
                                        0, 0, 0, 0, 1, 1, 1, 1,
                                        0, 0, 0, 0, 1, 1, 1, 1,
                                        0, 0, 0, 0, 1, 1, 1, 1]))

        self.assertTrue(np.allclose(i, [0, 0, 1, 1, 0, 0, 1, 1,
                                        0, 0, 1, 1, 0, 0, 1, 1,
                                        1, 1, 2, 2, 1, 1, 2, 2,
                                        1, 1, 2, 2, 1, 1, 2, 2]))
        
        self.assertTrue(np.allclose(j, [0, 1, 0, 1, 0, 1, 0, 1,
                                        1, 2, 1, 2, 1, 2, 1, 2,
                                        0, 1, 0, 1, 0, 1, 0, 1,
                                        1, 2, 1, 2, 1, 2, 1, 2]))


    def test_flatten(self):
        
        img = np.array([[[1, 2, 3], 
                         [4, 5, 6], 
                         [7, 8, 9]], 
                        [[-1, -2, -3], 
                         [-4, -5, -6], 
                         [-7, -8, -9]]])

        patch = utils.flatten(img, (2, 3, 3), (2, 2), (0, 0), (1, 1))

        self.assertTrue(np.allclose(patch, [1, 2, 4, 5, -1, -2, -4, -5, 
                                            2, 3, 5, 6, -2, -3, -5, -6,
                                            4, 5, 7, 8, -4, -5, -7, -8,
                                            5, 6, 8, 9, -5, -6, -8, -9]))

    def test_unflatten(self):
        
        col = np.array([[1, 1, 1, 1, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0]])
        img = utils.unflatten(col, (2, 3, 3), (2, 2), (0, 0), (1, 1))
        img_true = np.array([[[1, 1, 0], 
                              [1, 1, 0], 
                              [0, 0, 0]], 
                             [[0, 1, 1], 
                              [0, 2, 1], 
                              [0, 1, 0]]])
        self.assertTrue(np.allclose(img, img_true))

    def test_compute_momentum(self):    
        v0 = np.zeros([10])
        dx = np.arange(10)
        config = {
            'mu' : 0.9,
            'step_size' : 1
        }
        v1 = utils.compute_momentum(v0, dx, config)
        self.assertTrue(np.allclose(v1, [-0, -1, -2, -3, -4, -5, -6, -7, -8, -9]))

        dx = np.array([4, 3, 2, 1, 0, 4, 3, 2, 1, 0])
        
        v2 = utils.compute_momentum(v1, dx, config)
        self.assertTrue(np.allclose(v2, [-4, -3.9, -3.8, -3.7, -3.6, -8.5, -8.4, -8.3, -8.2, -8.1]))

if __name__ == "__main__":
    unittest.main()