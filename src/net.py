import time

class Net:
    def __init__(self, shape):
        self.shape = shape 
        self.layers = []
        
    def __repr__(self):
        return str(self.layers)

    def add_layer(self, layer):
        if layer.accept(self.shape) is True:
            self.shape = layer.shape
            self.layers.append(layer)
            return self
        else:
            return None


    def predict(self, x):
        y = x
        for layer in self.layers:
            y = layer.feed_forward(y)
        return y

    def learn(self, dy):
        dx = dy
        for layer in reversed(self.layers):
            dx = layer.feed_backward(dx)     

    def evaluate(self, yhat, y):
        return None, None

    def train_iteration(self, x, y):
        yhat = self.predict(x)

        loss, dy = self.evaluate(yhat, y)

        self.learn(dy)

        return loss

    def book_keeping_loss(self, loss):
        pass

    def fit(self, x, y, iteration):
        self.loss_history = [None] * iteration
        start = time.time()

        print('Training started...')
        for t in range(iteration):
            self.book_keeping_loss(self.train_iteration(x, y))
            
            time_remain = (time.time() - start) / (t + 1) * (iteration - t- 1)
            print('Iter  {0:4d} | Time Remain: {1:4.2f}'.format(t, time_remain), end='\r')  
        print('Training finished. took {0:4.2f} s'.format(time.time() - start))