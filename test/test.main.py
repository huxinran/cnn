import unittest
import sys
import numpy as np
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\')
from conv_layer import ConvLayer as L

class TestLayer(unittest.TestCase):

    def test_compute_output_len(self):
        self.assertTrue((L.get_pos(3, 2, 0, 1) == np.array([0, 1])).all())
        self.assertTrue((L.get_pos(7, 3, 0, 1) == np.array([0, 1, 2, 3, 4])).all())
        self.assertTrue((L.get_pos(5, 3, 1, 2) == np.array([-1, 1, 3])).all())

    def test_get_conv_index(self):
        index = L.init_conv_index([1, 3, 3], [2, 2], 0, 1)
        #self.assertEqual(, 1)

    def test_convert_pixel_to_conv(self):
        x = np.array([
            0.374, 0.950, 0.731, 
            0.598, 0.156, 0.155, 
            0.058, 0.866, 0.601
        ])
        conv_x_true = np.array([
            [0.374, 0.950, 0.598, 0.156],
            [0.950, 0.731, 0.156, 0.155],
            [0.598, 0.156, 0.058, 0.866],
            [0.156, 0.155, 0.866, 0.601]
        ])
        index = L.init_conv_index([1, 3, 3], [2, 2], 0, 1)
        conv_x = L.conv(x, index)
        self.assertTrue((conv_x == conv_x_true).all())

    def test_conv_layer_init(self):
        c = L([1, 3, 3], [2, 2], 3)

    def test_feed_forward(self):
        c = L([1, 3, 3], [2, 2], 2)
        c.w = np.array([
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 1]
        ])
        c.b = np.array([[0.2, -0.1]])

        x = np.array([
            [1, 2, 3, 
             4, 5, 6, 
             7, 8, 9], 
            [19, 18, 17, 
             16, 15, 14, 
             13, 12, 11]
        ])
        y = c.feed_forward(x)
        y_true = np.array([
            [1.2, 2.2, 4.2, 5.2, 4.9, 5.9, 7.9, 8.9],
            [19.2, 18.2, 16.2, 15.2, 14.9, 13.9, 11.9, 10.9]
        ])
        self.assertTrue((y == y_true).all())


if __name__ == "__main__":
    unittest.main()