import unittest
import sys
import numpy as np
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\')
from layer import Layer as L

class TestLayer(unittest.TestCase):

    def test_compute_output_len(self):
        self.assertTrue((L.get_pos(3, 2, 0, 1) == np.array([0, 1])).all())
        self.assertTrue((L.get_pos(7, 3, 0, 1) == np.array([0, 1, 2, 3, 4])).all())
        self.assertTrue((L.get_pos(5, 3, 1, 2) == np.array([-1, 1, 3])).all())

    def test_get_conv_index(self):
        p_shape = np.array([1, 3, 3])
        f_shape = np.array([1, 2, 2])
        index = L.get_conv_index(p_shape, f_shape, 0, 1)
        #print(index)
        #self.assertEqual(, 1)

    def test_convert_pixel_to_conv(self):
        p_shape = np.array([1, 3, 3])
        f_shape = np.array([1, 2, 2])
        pixel = np.random.random(9)
        conv = L.convert_pixel_to_conv(pixel, p_shape, f_shape, 0, 1)
        print(conv)
        #self.assertEqual(, 1)



if __name__ == "__main__":
    unittest.main()