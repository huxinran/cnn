import unittest
import numpy as np

import sys
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\')
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\layer\\')
from conv_layer import ConvLayer as Conv

class TestConvLayer(unittest.TestCase):
    def test_init(self):
        l = Conv([2, 2], 2)
        self.assertEqual(l.height_k, 2)
        self.assertEqual(l.width_k, 2)
        self.assertEqual(l.depth_out, 2)
        self.assertEqual(l.pad, 0)
        self.assertEqual(l.stride, 1)
        self.assertEqual(l.type, 'ConvLayer')

    def test_accept(self):
        l = Conv([2, 2], 2)
        l.accept([2, 3, 3])
        self.assertEqual(l.height_k, 2)
        self.assertEqual(l.width_k, 2)
        self.assertEqual(l.depth_out, 2)
        self.assertEqual(l.pad, 0)
        self.assertEqual(l.stride, 1)
        self.assertEqual(l.type, 'ConvLayer')

    def test_forward(self):

        l = Conv([2, 2], 2)
        l.accept([1, 3, 3])
        
        l.w = np.array([
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 1]
        ])
        l.b = np.array([0.0, 0.1])

        x = np.array([
             [[[1, 2, 3], 
               [4, 5, 6], 
               [7, 8, 9]]],
             [[[-1, -2, -3], 
               [-4, -5, -6], 
               [-7, -8, -9]]]
            ])

        y = l.forward(x)
        y_true = np.array([
            [1, 2, 4, 5, 5.1, 6.1, 8.1, 9.1],
            [-1, -2, -4, -5, -4.9, -5.9, -7.9, -8.9]
        ])

        self.assertTrue(np.allclose(y, y_true))

    def test_backward(self):
        l = Conv([2, 2], 2)
        l.accept([1, 3, 3])
        
        l.w = np.array([
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 1]
        ])

        l.b = np.array([0.0, 0.1])
        
        l.flat_x = np.array([[
            [1, 2, 3, 4],
            [11, 12, 13, 14],
            [21, 22, 23, 24],
            [31, 32, 33, 34]
        ]])

        dy = np.array([
            [1, 2, 3, 4, -4, -3, -2, -1],
        ])

        dx, dw, db = l.backward(dy)

        dx_true = np.array([[1, 2, 0, 3, 0, -3, 0, -2, -1]])

        dw_true = np.array([
            [210, -110],
            [220, -120],
            [230, -130],
            [240, -140]
        ])

        db_true = np.array([10, -10])

        self.assertTrue((dx == dx_true).all())
        self.assertTrue((dw == dw_true).all())
        self.assertTrue((db == db_true).all())


    def test_repr(self):
        l = Conv([2, 2], 2)
        l.accept([2, 3, 3])
        

if __name__ == "__main__":
    unittest.main()