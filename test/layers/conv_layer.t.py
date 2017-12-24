import unittest
import numpy as np

import sys
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\')

from conv_layer import ConvLayer as Conv

class TestConvLayer(unittest.TestCase):
    def test_init(self):
        l = Conv(2, 2, 2)
        self.assertEqual(l.height_k, 2)
        self.assertEqual(l.width_k, 2)
        self.assertEqual(l.depth_out, 2)
        self.assertEqual(l.pad, 0)
        self.assertEqual(l.stride, 1)
        self.assertEqual(l.type, 'ConvLayer')

    def test_accept(self):
        l = Conv(2, 2, 2)
        l.accept(2, 3, 3)
        self.assertEqual(l.height_k, 2)
        self.assertEqual(l.width_k, 2)
        self.assertEqual(l.depth_out, 2)
        self.assertEqual(l.pad, 0)
        self.assertEqual(l.stride, 1)
        self.assertEqual(l.type, 'ConvLayer')
        pass
    
    def test_repr(self):
        pass
    

if __name__ == "__main__":
    unittest.main()