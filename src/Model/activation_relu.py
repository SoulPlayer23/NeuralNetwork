import numpy as np

class Activation_ReLU:
    def forward(self, inputs):
        # if input > 0, return input, else return 0
        self.output = np.maximum(0 ,inputs)
        self.inputs = inputs
    
    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0