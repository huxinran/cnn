import numpy as np
from layer import FullyConnectedLayer as FC
from utils import softmax
from utils import plot
import time
import matplotlib.pyplot as plt 

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
        
        self.w = [] 

        for i in range(self.l):
            self.w.append(FC.init_weight(self.dim[i], self.dim[i + 1]))
        
    def __repr__(self):
        return 'NeuralNet with {0.din} to {0.dhidden} to {0.dout}'.format(self)

    def compute_output(self, input_):
        """
        given a input data, 
        return an output value and a list of data for each of the hidden layers
        """
        input_hidden = [None] * self.l
        input_hidden[0] = input_
        
        for i in range(self.l - 1):
            input_hidden[i + 1] = np.maximum(0, FC.fwd(input_hidden[i], self.w[i]))
        
        output = FC.fwd(input_hidden[-1], self.w[-1])
        return output, input_hidden

    def compute_loss(self, output, label):
        """
        given the output and true label, 
        return softmax cross entropy loss and gradient on output
        """
        prob = softmax(output)
        loss = -np.log(np.maximum(np.exp(-10), prob[np.arange(label.shape[0]), label]))
        d_output = prob
        d_output[np.arange(label.shape[0]), label] -= 1
        return loss, d_output

    def compute_gradient(self, d_output, input_hidden):
        """
        given the graident on output and data for hidden layer
        computer the gradient for hidden layer weights
        """
        d_w_hidden = [None] * self.l
        for i in reversed(range(self.l)):
            d_output, d_w_hidden[i] = FC.bwd(d_output, input_hidden[i], self.w[i])      
            d_output *= 1 * (input_hidden[i] > 0)

        # normalized the gradient by number of observations
        for d_w in d_w_hidden:
            d_w /= d_output.shape[0]

        return d_w_hidden

    def apply_graident(self, d_w, step_size=0.001, regularization=0):
        """
        given gradient, apply gradient on weight with stepsize adjustment and 
        regularization
        """
        for i in range(self.l):
            self.w[i] -= (d_w[i] * step_size + self.w[i] * regularization)

    def train_iteration(self, data, label, debug=False):
        """
        one iteration of learning
        """
        # forawrd feed, get y and x_hidden
        output, input_hidden = self.compute_output(data)
        
        # measure loss and gradient on y
        loss, d_output = self.compute_loss(output, label)
        
        # backprop, get gradient on weight
        d_w_hidden = self.compute_gradient(d_output, input_hidden)
        
        if debug:
            print('w = ', self.w)
            print("xhidden = ", input_hidden)
            print("y = ", output)
            print("dy = ", d_output)
            print("dw = ", d_w_hidden)

        return output, loss, d_w_hidden

    def train_test_split(self, data, label, test_pct=0.0):
        N = data.shape[0]
        train_size = np.ceil(N * (1 - test_pct)).astype(int)
        return data[0:train_size,:], label[0:train_size], data[train_size:,:], label[train_size:]

    def normalize(self, data):
        data = data - np.mean(data, axis=0)
        return data

    def fit(self, data, label, iteration, step_size=0.001, regularization=0.0, test_pct=0.0, debug=False):
        start = time.time()
        
        data = self.normalize(data)
        d_train, l_train, d_test, l_test = self.train_test_split(data, label, test_pct)
        
        for t in range(iteration):
            # computer gradient on weight
            output_train, loss_train, d_w = self.train_iteration(d_train, l_train, debug)
            
            # apply gradient on weight
            self.apply_graident(d_w, step_size, regularization)

            # all book keeping
            output_test, _ = self.compute_output(d_test)

            avg_loss_train = np.mean(loss_train)

            err_rate_train = np.mean(1 * (np.argmax(output_train, axis=1) != l_train))
            err_rate_test  = np.mean(1 * (np.argmax(output_test, axis=1) != l_test))

            time_remain = (time.time() - start) / (t + 1) * (iteration - t - 1)
            debug_str = 'Iter:{0:4d} | Time:{1:4.4f} | TrainErr:{2:4.4f} | Test Err:{3:4.4f} | Loss:{4:4.4f}'.format(t, time_remain, err_rate_train,err_rate_test,avg_loss_train)
            print(debug_str, end='\r')

        print('\n\nTime total : {0}'.format(time.time() - start))