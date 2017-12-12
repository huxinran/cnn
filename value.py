class Data:
    def __init__(self, val):
        self.val = val

    def forward(self):
        return self.val
    
    def backward(self, gradient):
        pass


class Variable:
    def __init__(self, val):
        self.val = val
    
    def forward(self):
        return self.val
    
    def backward(self, gradient):
        self.val += gradient


class Mul:
    def __init__(self, child):
        self.val = 1.0
        self.child = child
    
    def forward(self):
        self.val = 1.0
        for n in self.child:
            self.val *= n.forward()
        return self.val
    
    def backward(self, gradient):
        for n in self.child:
            n.backward(gradient * self.val / n.val)

class Add:
    def __init__(self, child):
        self.child = child
        self.val = 0.0

    def forward(self):
        self.val = 0.0
        for n in self.child:
            self.val += n.forward()
        return self.val

    def backward(self, gradient):
        for n in self.child:
            n.backward(gradient)


class Max:
    def __init__(self, child):
        self.child = child
        self.val = 0.0

    def forward(self):
        self.val = 0.0
        for n in self.child:
            self.val += n.forward()
        return self.val

    def backward(self, gradient):
        for n in self.child:
            n.backward(gradient)