import time
import numpy as np
import sys
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\')
import utils

class Net:
    def __init__(self, config):
        self.config = config
        self.shape = config['input_shape']
        self.layer = []
        
    def __repr__(self):
        return '\n'.join(['Layer {0} : {1}'.format(i, str(l)) for i, l in enumerate(self.layer)])
        
    def add(self, layer):
        if layer.accept(self.shape):
            self.shape = layer.shape
            self.layer.append(layer)
            return self
        else:
            raise Exception('Could not add {0} to the model'.format(layer))

    def forward(self, x):
        y = x
        for l in self.layer:
            y, cache = l.forward(y)
            l.cache = cache
        return y

    def backward(self, dy):
        dx = dy
        for l in reversed(self.layer):
            dx, dparam = l.backward(dx)
            l.dparam = dparam

    def learn(self):
        for l in self.layer:
            l.learn(l.dparam)

    def evaluate(self, y, y_true):
        p = utils.softmax(y)
        return utils.cross_entropy(p, y_true)

    def train_one_iteration(self, x, y_true):
        y = self.forward(x)
        
        loss, dy = self.evaluate(y, y_true)
        
        self.backward(dy)

        self.learn()

        return loss, y

    def book_keeping_loss(self, loss):
        avg_loss = np.mean(loss) 
        return 'loss = {0:4.2f}'.format(avg_loss) 

    def fit(self, x, y, iteration):
        self.loss_history = [None] * iteration
        start = time.time()

        print('Training started...')
        
        for t in range(iteration):
            if t % 10 == 0:
                self.config['step_size'] *= self.config['step_decay']

            loss, yfit = self.train_one_iteration(x, y)
     
            msg = self.book_keeping_loss(loss)
            
            err = np.mean(1 * (np.argmax(yfit, axis=1) != y))
            
            time_remain = (time.time() - start) / (t + 1) * (iteration - t- 1)

            print('Iter{0:4d}| {2} | {3:4.2f} | Time Remain: {1:4.1f} '.format(t, time_remain, msg, err))  
        
        print('Training finished. took {0:4.2f} s'.format(time.time() - start))