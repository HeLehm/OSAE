import torch
from typing import Dict


def mean_cosine_similarity(x, x_hat):
    """
    Compute the mean cosine similarity between x and x_hat.
    """
    return torch.nn.functional.cosine_similarity(x_hat, x).mean()


def dead_neurons_batch(c: torch.Tensor) -> torch.Tensor:
    """
    Given a batch of activations, return a mask of dead neurons.
    """
    return c.abs().mean(dim=0) == 0


class DeadNeuronDetector:
    def __init__(self):
        self.dead_neurons_counter = None
        self.sample_counter = 0

    def on_batch(self, c):
        """
        Given a batch of activations, update the mask of dead neurons.
        """
        self.sample_counter += 1
        mask = dead_neurons_batch(c).to(torch.float32)
        if self.dead_neurons_counter is None:
            self.dead_neurons_counter = mask
        else:
            self.dead_neurons_counter += mask

    def on_epoch_end(self) -> Dict[int, float]:
        """
        Return a dict, where keys are indices of neurons
        and keys are the proportion of batches, where the neuron was dead.
        I.e. if values is 1.0 for a neuron, it was dead in all batches.
        NOTE: resets the instance.
        """
        dead_neurons = self.dead_neurons_counter / self.sample_counter
        self.reset()

        indices = dead_neurons.argsort(descending=True)
        values = dead_neurons[indices]

        # count where values == 1.0
        dead_count = (values == 1.0).sum().item()
        return {i.item(): v.item() for i, v in zip(indices, values)}, dead_count

    def reset(self):
        """
        Reset the mask of dead neurons.
        """
        self.dead_neurons = None
        self.sample_counter = 0
