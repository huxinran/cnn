"""
neural net class
"""
import time
import numpy as np
from src.layer import FullyConnectedLayer as FC
from src.utils import softmax
from src.utils import train_test_split
from src.utils import normalize
from src.utils import compute_loss

class NeuralNet:
    '''
    Neural Net class consists of k fully connected layers which takes an
    array of data of size [din] and classify them into [dout] labels

    '''
    def __init__(self, dim):
        '''
        dim  : an array of dimensions
        [din,       dh1, dh2, ..., dhk, dout]
         input -> | hidden layers    | -> output
        '''
        self.dim = dim
        self.l = len(self.dim) - 1
        self.w = [None] * self.l
        self.b = [None] * self.l
        for i in range(self.l):
            self.w[i], self.b[i] = FC.init_weight(self.dim[i], self.dim[i + 1])

    def compute_output(self, input_):
        """
        given a input data,
        return an output value and a list of data for each of the hidden layers
        """
        input_hidden = [None] * self.l
        input_hidden[0] = input_
        for i in range(self.l - 1):
            input_hidden[i + 1] = np.maximum(0, FC.fwd(input_hidden[i], self.w[i], self.b[i]))
        output = FC.fwd(input_hidden[-1], self.w[-1], self.b[-1])
        return output, input_hidden



    def compute_gradient(self, g_output, input_hidden):
        """
        given the graident on output and data for hidden layer
        computer the gradient for hidden layer weights
        """
        g_w_hidden = [None] * self.l
        g_b_hidden = [None] * self.l
        for i in reversed(range(self.l)):
            g_output, g_w_hidden[i], g_b_hidden[i] = FC.bwd(g_output, input_hidden[i], self.w[i])
            g_output *= 1 * (input_hidden[i] > 0)
        # normalized the gradient by number of observations
        for i in range(self.l):
            g_w_hidden[i] /= g_output.shape[0]
            g_b_hidden[i] /= g_output.shape[0]

        return g_w_hidden, g_b_hidden

    def apply_graident(self, g_w, g_b, step_size=0.001, regularization=0):
        """
        given gradient, apply gradient on weight with stepsize adjustment and
        regularization
        """
        for i in range(self.l):
            self.w[i] -= (g_w[i] * step_size + self.w[i] * regularization)
            self.b[i] -= (g_b[i] * step_size + self.b[i] * regularization)

    def train_iteration(self, data, label, debug=1):
        """
        one iteration of learning
        """
        # forawrd feed, get y and x_hidden
        output, input_hidden = self.compute_output(data)
        # measure loss and gradient on y
        loss, g_output = compute_loss(output, label)
        # backprop, get gradient on weight
        g_w_hidden, g_b_hidden = self.compute_gradient(g_output, input_hidden)
        if debug:
            debugStr = 'w={0} \n xhidden={1} \n y={2} \n dy={3} \n dw={4}'.format(self.w, input_hidden, output, g_output, g_w_hidden)
            print(debugStr)

        return output, loss, g_w_hidden, g_b_hidden

    def fit(self, data, label, iteration=10, step_size=0.001, regularization=0.0, test_pct=0.0, debug=1):
        """
        fit base on data and label
        """
        start = time.time()
        data = normalize(data)
        d_train, l_train, d_test, l_test = train_test_split(data, label, test_pct)
        for t in range(iteration):
            # computer gradient on weight
            output_train, loss_train, d_w, d_b = self.train_iteration(d_train, l_train, debug)
            # apply gradient on weight
            self.apply_graident(d_w, d_b, step_size, regularization)
            # all book keeping
            output_test, _ = self.compute_output(d_test)
            avg_loss_train = np.mean(loss_train)
            err_rate_train = np.mean(1 * (np.argmax(output_train, axis=1) != l_train))
            err_rate_test = np.mean(1 * (np.argmax(output_test, axis=1) != l_test))
            time_remain = (time.time() - start) / (t + 1) * (iteration - t - 1)
            
            if debug >= 0:
                debug_str = 'Iter:{0:4d} | Time:{1:4.2f} | TrainErr:{2:4.2f} | Test Err:{3:4.2f} | Loss:{4:4.2f}'.format(t, time_remain, err_rate_train, err_rate_test, avg_loss_train)
                print(debug_str, end='\r')

        print('\n\nTime total : {0}'.format(time.time() - start))
