class Neuron:
    def __init__(self, inputDim, activateFunc):
        self.inputDim = inputDim
        self.activateFunc = activateFunc
        self.weight = np.zeros(inputDim + 1, 1)
        self.value = 0.0
        

    def forward(input):
        self.value = max(0, np.inner(input, self.weight))
        return self.value
    
    def backward(gradient):
        if 

    
