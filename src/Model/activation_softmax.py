import numpy as np

class Activation_Softmax:
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Get unormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        # Calculate the softmax of the input z
        # Formula: σ(z) = exp(z) / Σ exp(z)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    
    def  backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate the Jacobian matrix of the output and 
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample graidents
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)