import numpy as np

class Loss:
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss: data_loss = (1/n) * Σ(sample_losses)
        data_loss = np.mean(sample_losses)
        return data_loss