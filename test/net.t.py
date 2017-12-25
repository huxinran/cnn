import unittest
import numpy as np

import sys
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\')
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\layers')

from net import Net
from fc_layer import FullyConnectedLayer as FC

class TestNet(unittest.TestCase):
    def test_init(self):
        n = Net(np.array([3, 5, 5]))

    def test_add_layer(self):
        n = Net(np.array([3, 5, 5]))
        l = FC(10)
        n.add_layer(l)

    def test_train_iteration(self):
        n = Net(np.array([3, 5, 5]))
        l = FC(10)
        n.add_layer(l)
        x = np.random.random([1, 3, 5, 5]).reshape(1, -1)
        y = np.array([0])
        loss = n.train_one_iteration(x, y)
        print(loss)
    
    def test_fit(self):
        n = Net(np.array([3, 5, 5]))
        l = FC(10)
        n.add_layer(l)
        x = np.random.random([1, 3, 5, 5]).reshape(1, -1)
        y = np.array([0])
        n.fit(x, y, 10)
        

if __name__ == "__main__":
    unittest.main()