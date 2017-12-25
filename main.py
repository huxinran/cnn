"""
main
"""
import numpy as np
import matplotlib.pyplot as plt
from src.net import Net
from data.data import mnist
plt.ion()


def main():
    """
    main func
    """
    
    (data, label) = mnist()
    data = np.array(data)
    label = np.array(label)
    din = data[0].size
    dhidden = 100
    dout = np.unique(label).size
    step_size = 1
    iteration = 1000
    regularization = 0.0001
    debug = False
    np.random.seed(4)


if __name__ == "__main__":
    main()
