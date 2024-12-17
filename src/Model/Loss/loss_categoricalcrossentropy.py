import numpy as np
from .loss import Loss

class Loss_CategoricalLossEntropy(Loss):
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses: categorical cross-entropy loss formula
        # L = -∑(y_true * log(y_pred))
        # where y_true is one-hot encoded
        # Here, we use the property of one-hot encoding: y_true * log(y_pred) = log(y_pred[y_true])
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
            
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        # Calculate gradient: derivative of categorical cross-entropy loss formula
        # dL/dy_pred = -y_true / y_pred
        self.dinputs = -y_true / dvalues

        # Normalize gradient
        self.dinputs = self.dinputs / samples
