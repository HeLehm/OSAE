import torch
from typing import Dict


def mean_pairwise_cosine_similarity(x, x_hat):
    """
    Compute the mean cosine similarity between x and x_hat.
    """
    return torch.nn.functional.cosine_similarity(x_hat, x).mean()


def mean_max_cosine_similarity(x):
    """
    Compute the mean of the maximum cosine similarity between all pairs of vectors in x.
    """
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    cos_table = x @ x.T
    # exclude to self
    cos_table = cos_table - torch.eye(cos_table.shape[0], device=cos_table.device)
    return cos_table.max(dim=1).values.mean()


def dead_neurons_batch(c: torch.Tensor, dead_1=True) -> torch.Tensor:
    """
    Given a batch of activations, return a mask of dead neurons.
    """
    if dead_1:
        return c.abs().mean(dim=0) == 0
    return c.abs().mean(dim=0) != 0


class DeadNeuronDetector:
    def __init__(self):
        self.dead_neurons_counter = None
        self.sample_counter = 0

    def on_batch(self, c):
        """
        Given a batch of activations, update the mask of dead neurons.
        """
        self.sample_counter += 1
        mask = dead_neurons_batch(c, dead_1=True).to(torch.float32)
        if self.dead_neurons_counter is None:
            self.dead_neurons_counter = mask
        else:
            self.dead_neurons_counter += mask

    def on_epoch_end(self, reset=True) -> Dict[int, float]:
        """
        Return a dict, where keys are indices of neurons
        and keys are the proportion of batches, where the neuron was dead.
        I.e. if values is 1.0 for a neuron, it was dead in all batches.
        NOTE: resets the instance.
        """
        dead_neurons = self.dead_neurons_counter / self.sample_counter
        if reset:
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
        self.dead_neurons_counter = None
        self.sample_counter = 0


class SlidingWindowDeadNeuronTracker:
    def __init__(
        self,
        max_len: int,
        embedding_dim: int,
    ) -> None:
        assert max_len > 1, "max_len must be greater than 1"
        self.max_len = max_len
        # 0 means inactive, 1 means active
        self.neuron_activity = torch.ones(max_len, embedding_dim, dtype=torch.bool)

    @torch.no_grad()
    def on_batch(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Update the activity status of neurons based on the current batch of activations.

        Parameters:
        -----------
        activations: torch.Tensor
            A tensor containing the activations of neurons in the current batch.

        Returns:
        --------
        inactive_period: torch.Tensor
            A tensor of shape (embedding_dim,) where each element represents
            the inactive period of the neuron, normalized between 0 and 1.
        """
        mask = dead_neurons_batch(activations, False).to(torch.bool).to("cpu")

        # Roll the tensor to the left so that the newest element is at the end
        self.neuron_activity = torch.roll(self.neuron_activity, -1, dims=0)
        # Update the newest element with the current mask
        self.neuron_activity[-1, :] = mask

        return self.get_inactive_period()

    def get_inactive_period(self) -> torch.Tensor:
        """
        Calculate the inactive period for each neuron.

        Returns:
        --------
        inactive_period: torch.Tensor
            A tensor of shape (embedding_dim,) where each element represents
            the inactive period of the neuron. 0.0 means the neuron was active in the last batch,
            and 1.0 means the neuron was inactive in the last max_len batches.
        """
        steps_since_active = self._get_last_active_index().to(torch.float32)
        # Normalize the steps since last active to a range between 0 and 1
        steps_since_active.div_(self.max_len).mul_(-1.0).add_(1.0)
        return steps_since_active

    def _get_last_active_index(self) -> torch.Tensor:
        """
        Calculate the index of the last time a neuron was active.

        Returns:
        --------
        steps_since_active: torch.Tensor
            A tensor of shape (embedding_dim,) where each element represents
            the number of steps since the neuron was last active.
        """
        result_tensor = torch.zeros_like(self.neuron_activity[0, :]).to(torch.long)

        for i in range(self.max_len):
            mask = self.neuron_activity[i, :]
            result_tensor[mask] = i + 1

        return result_tensor

    def reset(self):
        """
        Reset the activity status of all neurons.
        """
        self.neuron_activity = torch.ones_like(self.neuron_activity)
