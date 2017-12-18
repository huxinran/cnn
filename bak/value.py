class Constant:
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
        if gradient == 0:
            return

        for i in range(len(self.child)):
            m = 1.0
            for j in range(len(self.child)):
                if i != j:
                    m *= self.child[j].val
            self.child[i].backward(gradient * m)

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
        self.maxChild = None
        self.val = None

    def forward(self):
        self.maxChild = None
        self.val = None
        for n in self.child:
            if self.val is None or self.val < n.forward():
                self.maxChild = n
                self.val = n.val
        
        return self.val

    def backward(self, gradient):
        self.maxChild.backward(gradient)


class ReLU:
    def __init__(self, src):
        self.src = src
        self.val = None

    def forward(self):
        v = self.src.forward()
        if v > 0.0:
            self.val = v
        else:
            self.val = 0.0
        
        return self.val
    
    def backward(self, dz)
        if self.val > 0.0:
            self.src.backward(dz)



