import unittest
import numpy as np

import sys
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\')
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\layer\\')
from conv import ConvLayer as Conv

class TestConvLayer(unittest.TestCase):

    def test_init(self):
        config = {
            'kernel_shape' : [2, 2]
          , 'output_depth' : 2
          , 'pad' : [0, 0]
          , 'stride' : [1, 1]
        }

        l = Conv(config)
        self.assertEqual(l.hk, 2)
        self.assertEqual(l.wk, 2)
        self.assertEqual(l.dout, 2)
        self.assertEqual(l.pad[0], 0)
        self.assertEqual(l.pad[1], 0)
        self.assertEqual(l.stride[0], 1)
        self.assertEqual(l.stride[1], 1)
        self.assertEqual(l.type, 'Conv')

    def test_accept(self):
        config = {
            'kernel_shape' : [2, 2]
          , 'output_depth' : 2
          , 'pad' : [0, 0]
          , 'stride' : [1, 1]
        }

        l = Conv(config)

        l.accept([2, 3, 3])
        self.assertEqual(l.hk, 2)
        self.assertEqual(l.wk, 2)
        self.assertEqual(l.dout, 2)
        self.assertEqual(l.pad[0], 0)
        self.assertEqual(l.pad[1], 0)
        self.assertEqual(l.stride[0], 1)
        self.assertEqual(l.stride[1], 1)
        self.assertEqual(l.type, 'Conv')

    def test_forward(self):
        config = {
            'kernel_shape' : [2, 2]
          , 'output_depth' : 2
          , 'pad' : [0, 0]
          , 'stride' : [1, 1]
        }

        l = Conv(config)
        l.accept([1, 3, 3])
        
        l.param['w'] = np.array([
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 1]
        ])
        l.param['b'] = np.array([0.0, 0.1])

        x = np.array([
             [[[1, 2, 3], 
               [4, 5, 6], 
               [7, 8, 9]]],
             [[[-1, -2, -3], 
               [-4, -5, -6], 
               [-7, -8, -9]]]
            ])

        y, cache = l.forward(x)
        y_true = np.array([
            [1, 2, 4, 5, 5.1, 6.1, 8.1, 9.1],
            [-1, -2, -4, -5, -4.9, -5.9, -7.9, -8.9]
        ])

        self.assertTrue(np.allclose(y, y_true))

    def test_backward(self):
        config = {
            'kernel_shape' : [2, 2]
          , 'output_depth' : 2
          , 'pad' : [0, 0]
          , 'stride' : [1, 1]
        }

        l = Conv(config)
        l.accept([1, 3, 3])
        
        l.param['w'] = np.array([
            [1.0, 0],
            [0, 0],
            [0, 0],
            [0, 1]
        ])

        l.param['b'] = np.array([[0.0, 0.1]])
        
        l.cache['flatten_x'] = np.array([[
            [1, 2, 3, 4],
            [11, 12, 13, 14],
            [21, 22, 23, 24],
            [31, 32, 33, 34]
        ]])

        dy = np.array([
            [1, 2, 3, 4, -4, -3, -2, -1],
        ])

        dx, dparam = l.backward(dy)

        dx_true = np.array([[1, 2, 0, 3, 0, -3, 0, -2, -1]])

        w_true = np.array([
            [210, -110],
            [220, -120],
            [230, -130],
            [240, -140]
        ])

        b_true = np.array([10, -10])
        self.assertTrue(np.allclose(dx, dx_true))
        self.assertTrue(np.allclose(dparam['w'], w_true))
        self.assertTrue(np.allclose(dparam['b'], b_true))


    def test_repr(self):
        config = {
            'kernel_shape' : [2, 2]
          , 'output_depth' : 2
          , 'pad' : [0, 0]
          , 'stride' : [1, 1]
        }

        l = Conv(config)
        l.accept([2, 3, 3])
        

if __name__ == "__main__":
    unittest.main()