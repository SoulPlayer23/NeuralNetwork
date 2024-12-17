class Neuron:
    def __init__(self, bias, weights, input, output):
        self.bias = bias
        self.weights = weights
        self.input = input
        self.output = output

    def fowrward_pass(self):
        pass
    
singleNeuron = Neuron(1, [1, 1], [1, 1], 1)