import time
import utils

class Net:
    def __init__(self, shape):
        self.shape = shape 
        self.layer = []
        
    def __repr__(self):
        return str(self.layer)

    def add_layer(self, layer):
        if layer.accept(self.shape) is True:
            self.shape = layer.shape
            self.layer.append(layer)
            return self
        else:
            return None


    def forward(self, x):
        y = x
        for layer in self.layer:
            y = layer.forward(y)
        return y

    def backward(self, dy):
        dx = dy
        for layer in reversed(self.layer):
            dx = layer.backward(dx)     

    def evaluate(self, y, y_true):
        p = utils.softmax(y)
        return utils.cross_entropy(p, y_true)

    def train_one_iteration(self, x, y_true):
        y = self.forward(x)
        
        loss, dy = self.evaluate(y, y_true)

        self.backward(dy)

        return loss

    def book_keeping_loss(self, loss):
        pass

    def fit(self, x, y, iteration):
        self.loss_history = [None] * iteration
        start = time.time()

        print('Training started...')
        for t in range(iteration):
            self.book_keeping_loss(self.train_one_iteration(x, y))
            
            time_remain = (time.time() - start) / (t + 1) * (iteration - t- 1)
            print('Iter  {0:4d} | Time Remain: {1:4.2f}'.format(t, time_remain), end='\r')  
        print('Training finished. took {0:4.2f} s'.format(time.time() - start))