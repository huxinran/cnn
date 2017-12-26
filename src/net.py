import time
import numpy as np
import sys
sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\')
import utils

class Net:
    def __init__(self, shape):
        self.shape = shape 
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
            y = l.forward(y)
        return y

    def backward(self, dy):
        dx = dy
        for l in reversed(self.layer):
            dx = l.backward(dx)

    def update(self, config):
        for l in self.layer:
            l.update(config)

    def evaluate(self, y, y_true):
        p = utils.softmax(y)
        return utils.cross_entropy(p, y_true)

    def train_one_iteration(self, x, y_true):
        y = self.forward(x)
        
        loss, dy = self.evaluate(y, y_true)
        
        self.backward(dy)

        config = {
            'step_size' : 0.00001
          , 'mu'        : 0.9
        }
        self.update(config)

        return loss, y

    def book_keeping_loss(self, loss):
        avg_loss = np.mean(loss) 
        return 'avg Loss = {0:4.2f}'.format(avg_loss) 

    def fit(self, x, y, iteration):
        self.loss_history = [None] * iteration
        start = time.time()

        print('Training started...')
        
        for t in range(iteration):
            loss, yfit = self.train_one_iteration(x, y)
            #print(loss)
            msg = self.book_keeping_loss(loss)
            
            err = np.mean(1 * (np.argmax(yfit, axis=1) != y))
            
            time_remain = (time.time() - start) / (t + 1) * (iteration - t- 1)

            print('Iter  {0:4d} | Time Remain: {1:4.2f} | {2} | {3}'.format(t, time_remain, msg, err))  
        
        print('Training finished. took {0:4.2f} s'.format(time.time() - start))