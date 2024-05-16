from .sae import TiedSparseAutoEncoder
import torch


def mean_cosine_similarity(x, x_hat):
    """
    Compute the mean cosine similarity between x and x_hat.
    """
    return torch.nn.functional.cosine_similarity(x_hat, x).mean()

def dead_neurons_batch(c: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    Given a batch of activations, return a mask of dead neurons.
    """
    return c.abs().mean(dim=0) < threshold

class DeadNeuronDetector():

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self.dead_neurons = None


    def on_batch(self,c):
        """
        Given a batch of activations, update the mask of dead neurons.
        """
        mask = dead_neurons_batch(c, self.threshold)
        if self.dead_neurons is None:
            self.dead_neurons = mask
        else:
            self.dead_neurons = self.dead_neurons | mask

        
    def on_epoch_end(self) -> torch.Tensor:
        """
        Return the mask of dead neurons.
        """
        return self.dead_neurons
    
    def reset(self):
        """
        Reset the mask of dead neurons.
        """
        self.dead_neurons = None
        