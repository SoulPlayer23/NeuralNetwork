import numpy as np
from Model.activation_relu import Activation_ReLU
from Model.activation_softmax_loss_categoricalcrossentropy import Activation_Softmax_Loss_CategoricalCrossEntropy
from Model.layer_dense import Layer_dense
import nnfs
from nnfs.datasets import spiral_data

def main():
    nnfs.init()

    # Create dataset
    x, y = spiral_data(samples=100, classes=3)

    # Create Dense layer with 2 input features and 3 output values
    dense1 = Layer_dense(2, 3)
    # Create ReLU activation (to be used with Dense layer):
    activation1 = Activation_ReLU()
    # Create second Dense layer with 3 input features (as we take output
    # of previous layer here) and 3 output values (output values)
    dense2 = Layer_dense(3, 3)
    # Create Softmax classifier's combined loss and activation
    loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
    # Perform a forward pass of our training data through this layer
    dense1.forward(x)
    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)

    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)
    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss = loss_activation.forward(dense2.output, y)

    # Let's see output of the first few samples:
    print(loss_activation.output[:5])
    # Print loss value
    print('loss:', loss)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if(len(y.shape) == 2):
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    # Print accuracy
    print('acc:', accuracy)

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Print gradients
    print(dense1.dweights)
    print(dense1.dbiases)
    print(dense2.dweights)
    print(dense2.dbiases)

if __name__ == "__main__":
    main()