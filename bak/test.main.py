import unittest
import sys
import numpy as np
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\')
from conv_layer import ConvLayer as L
from conv_layer import CNN as CNN
from utils import mnist 
data, label = mnist()
        

class TestLayer(unittest.TestCase):

    def test_get_pos(self):
        self.assertTrue((L.get_pos(3, 2, 0, 1) == np.array([0, 1])).all())
        self.assertTrue((L.get_pos(7, 3, 0, 1) == np.array([0, 1, 2, 3, 4])).all())
        self.assertTrue((L.get_pos(5, 3, 1, 2) == np.array([-1, 1, 3])).all())

    def test_init_conv_index(self):
        index = L.init_conv_index([1, 3, 3], [2, 2], 0, 1)
        index_true= np.array([
            [0, 1, 3, 4],
            [1, 2, 4, 5],
            [3, 4, 6, 7],
            [4, 5, 7, 8]
        ])
        self.assertTrue((index == index_true).all())

    def test_flat2conv(self):
        flat = np.array([
            0.0, 0.1, 0.2, 
            1.0, 1.1, 1.2, 
            2.0, 2.1, 2.2
        ])
        
        index = L.init_conv_index([1, 3, 3], [2, 2], 0, 1)

        conv = L.flat2conv(flat, index)
        
        conv_true = np.array([
            [0.0, 0.1, 1.0, 1.1],
            [0.1, 0.2, 1.1, 1.2],
            [1.0, 1.1, 2.0, 2.1],
            [1.1, 1.2, 2.1, 2.2]
        ])
        self.assertTrue((conv == conv_true).all())
    
    def test_conv2flat(self):
        conv = np.array([
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [-1, -1, -1, -1],
            [-2, -2, -2, -2]
        ])
        index = L.init_conv_index([1, 3, 3], [2, 2], 0, 1)

        flat = L.conv2flat(conv, index)
        flat_true = np.array([
            1, 3, 2, 
            0, 0, 0, 
            -1, -3, -2
        ])
        self.assertTrue((flat == flat_true).all())



    def test_fwd(self):
        x1 = np.array([
            1, 2, 3, 
            4, 5, 6, 
            7, 8, 9 
        ])

        x2 = np.array([
            19, 18, 17, 
            16, 15, 14, 
            13, 12, 11
        ])
        
        w = np.array([
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 1]
        ])
        b = np.array([[0.2, -0.1]])
        index = L.init_conv_index([1, 3, 3], [2, 2], 0, 1)

        y1, xconv1 = L.fwd(x1, w, b, index)
        y2, xconv2 = L.fwd(x2, w, b, index)
        y1_true = np.array([1.2, 2.2, 4.2, 5.2, 4.9, 5.9, 7.9, 8.9])
        y2_true = np.array([19.2, 18.2, 16.2, 15.2, 14.9, 13.9, 11.9, 10.9])
        
        self.assertTrue((y1 == y1_true).all())
        self.assertTrue((y2 == y2_true).all())

    def test_bwd(self):
        
        xconv = np.array([
            [1, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -2]
        ])

        w = np.array([
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 1]
        ])
        b = np.array([[0.2, -0.1]])
        index = L.init_conv_index([1, 3, 3], [2, 2], 0, 1)
        
        dy = np.array([
            1, 2, 
            3, 4, 
            -4, -3, 
            -2, -1
        ])

        dx, dw, db = L.bwd(dy, xconv, w, index)

        #===========
        dx_true = np.array([
            1, 2, 0, 
            3, 0, -3, 
            0, -2, -1 
        ])
        
        dw_true = np.array([
            [1, -4],
            [4, -6],
            [-3, 2],
            [-8, 2]
        ])
        
        db_true = np.array([[10, -10]])

        self.assertTrue((dx == dx_true).all())
        self.assertTrue((dw == dw_true).all())
        self.assertTrue((db == db_true).all())

    def test_init(self):
        l = L([1, 3, 3], [2, 2], 2, 0, 1)
        l = L([1, 3, 3], [2, 2], 2)       

    def test_feed_forward(self):
        l = L([1, 3, 3], [2, 2], 2, 0, 1)
        l.w = np.array([
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 1]
        ])
        l.b = np.array([0.0, 0.1])

        x = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [-1, -2, -3, -4, -5, -6, -7, -8, -9] 
        ])

        y = l.feed_forward(x)
        y_true = np.array([
            [1, 2, 4, 5, 5.1, 6.1, 8.1, 9.1],
            [-1, -2, -4, -5, -4.9, -5.9, -7.9, -8.9]
        ])

        self.assertTrue((y == y_true).all())

    def test_feed_backward(self):
        l = L([1, 3, 3], [2, 2], 2, 0, 1)
        l.w = np.array([
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 1]
        ])
        l.b = np.array([0.0, 0.1])
        
        l.xconv = np.array([[
            [1, 2, 3, 4],
            [11, 12, 13, 14],
            [21, 22, 23, 24],
            [31, 32, 33, 34]
        ]])


        dy = np.array([
            [1, 2, 3, 4, -4, -3, -2, -1],
        ])

        dx, dw, db = l.feed_backward(dy)

        dx_true = np.array([
            [1, 2, 0, 3, 0, -3, 0, -2, -1]
        ])

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

    def test_cnn_init(self):
        c = CNN()

    def test_cnn_train_iteration(self):
        c = CNN()
        y = c.train_iteration(data[0:10, :], label[0:10])

    def test_cnn_fit(self):
        c = CNN()
        c.fit(data[0:1000, :], label[0:1000], 100)



if __name__ == "__main__":
    unittest.main()