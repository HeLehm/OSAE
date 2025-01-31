import os

# make sure an openai api key is set
if "OPENAI_API_KEY" not in os.environ:
    # warn the user and set the key to a fake
    print("OPENAI_API_KEY not set, setting to fake key")
    os.environ["OPENAI_API_KEY"] = "fake_key"

import torch
import heapq
from torch import nn
from tqdm import tqdm
from collections import defaultdict
from typing import Optional, Union, List


from neuron_explainer.activations.activations import (
    ActivationRecord,
    NeuronId,
)

from ..act_dataset import ActivationDataset


from .dataclasses import NeuronRecord, NeuronRecords


# Define comparison functions for ActivationRecord
def activation_record_lt(self: ActivationRecord, other: ActivationRecord):
    # Compare based on the maximum activation value
    return max(self.activations) < max(other.activations)


# Monkey patch the comparison functions onto ActivationRecord
ActivationRecord.__lt__ = activation_record_lt


def activation_record_iter(
    activation_ds: ActivationDataset,
    sae: nn.Module,
    idx_filter: Optional[Union[int, List[int]]] = None,
    tqdm_desc: Optional[str] = None,
):
    """
    Yield ActivationRecords for sae and activation_ds.
    NOTE: fragment length should be conrolled by the activation dataset.

    Parameters
    ----------
    activation_ds : ActivationDataset
        ActivationDataset object.
    sae : nn.Module
        SparseAutoencoder object.
        forward should take x and return x_hat and c.
    idx_filter : Optional[Union[int,List[int]]]
        If None, iterate over all neurons.
        If int, iterate over the specified neuron.
        If List[int], iterate over the specified neurons.
        Default is None.

    Yields
    ------
    Tuple[int, ActivationRecord]
        Tuple of neuron index and ActivationRecord.
    """

    activation_ds.flatten_sequence = False
    device = next(sae.parameters()).device

    _iter = range(len(activation_ds))
    if tqdm_desc is not None:
        _iter = tqdm(_iter, desc=tqdm_desc)

    for i in _iter:
        x, corresponding_texts = activation_ds[i]
        x = x.to(device)

        # shape of c: (max_seq_length, num_hidden_units)
        c = sae.encode(x)

        # make a mask based on weather the text is ""
        mask = torch.tensor(
            [len(text) > 0 for text in corresponding_texts],
            dtype=torch.bool,
            device=device,
        )

        masked_corresponding_texts = [
            text for text, m in zip(corresponding_texts, mask) if m
        ]

        _iter = None
        if idx_filter is None:
            _iter = range(c.shape[-1])
        elif isinstance(idx_filter, int):
            _iter = [idx_filter]
        else:
            _iter = idx_filter

        for neuron_idx in _iter:
            neuron_activation = c[mask, neuron_idx].cpu().tolist()
            activation_record = ActivationRecord(
                activations=neuron_activation,
                tokens=masked_corresponding_texts,
            )

            yield neuron_idx, activation_record


@torch.no_grad()
def generate_neuron_records(
    activation_ds: ActivationDataset,
    sae: nn.Module,
    top_k: int = 20,
    idx_filter: Optional[Union[int, List[int]]] = None,
    layer_index: int = 0,
) -> NeuronRecords:
    """
    Generate NeuronRecord objects from the activation dataset and the model.
    NOTE: fragment length should be controlled by the activation dataset.
    OpeanAI seems to use 64 for max_seq_length.

    Parameters
    ----------
    activation_ds : ActivationDataset
        ActivationDataset object.
    sae : nn.Module
        SparseAutoencoder object.
    top_k : int, optional
        Number of top activations to keep for each neuron. Default is 20.
        Note that this is a max, if there are less than top_k activations, all will be kept.
    idx_filter : Optional[Union[int,List[int]]], optional
        If None, generate records for all neurons.
        If int, generate records for the specified neuron.
        If List[int], generate records for the specified neurons.
        Default is None.
    layer_index : int
        Only used for NeuronId.
        Layer index of the layer we are interested in for storage purposes.

    Returns
    -------
    NeuronRecords
        NeuronRecords.records: List of NeuronRecord objects.
            where NeuronRecord is a named tuple with fields:
                - neuron_id: NeuronId (nor containing neuron idx and layer idx)
                - most_positive_activation_records: List[ActivationRecord] (with len top_k)
                - random_sample: List[ActivationRecord] (with len top_k)
    """

    # To store top-k activation records for each neuron using a min-heap
    neuron_activations = defaultdict(list)
    # To store random samples for each neuron
    random_samples = defaultdict(list)
    # count how many times a neuron is activated
    neuron_activation_counts = {}

    # Iterate over activation records
    for i, (neuron_idx, activation_record) in enumerate(
        activation_record_iter(
            activation_ds, sae, idx_filter, tqdm_desc="Generating Neuron Records"
        )
    ):
        # chekc if neuron has been seen before
        if neuron_idx not in neuron_activation_counts:
            # initialize the neuron, so we can later create an empty neuron record
            neuron_activation_counts[neuron_idx] = 0

        # if a neuron is not activated, we do not need to do any claculatiosn with it
        if max(activation_record.activations) <= 0:
            continue

        # Increment the activation count for the neuron
        neuron_activation_counts[neuron_idx] += 1

        # Use a heap to maintain the top_k most positive activations
        # activation records will be compared based on the maximum activation value
        # see monkey patched __lt__ method in ActivationRecord
        if len(neuron_activations[neuron_idx]) < top_k:
            heapq.heappush(neuron_activations[neuron_idx], activation_record)
        else:
            heapq.heappushpop(neuron_activations[neuron_idx], activation_record)

        # Store random samples
        if len(random_samples[neuron_idx]) < top_k:
            random_samples[neuron_idx].append(activation_record)
        elif torch.rand(1).item() < 1 / (i + 1):
            # ranomly replace an element in the random sample
            random_samples[neuron_idx][torch.randint(top_k, (1,)).item()] = (
                activation_record
            )

    neuron_records = []
    for neuron_idx, activations_heap in neuron_activations.items():
        # Extract the activation records from the heap and sort them in descending order
        most_positive_activation_records = list(sorted(activations_heap, reverse=True))
        # get tre right random_sample
        random_sample = random_samples[neuron_idx]

        # Create a NeuronRecord object
        neuron_record = NeuronRecord(
            neuron_id=NeuronId(neuron_index=neuron_idx, layer_index=layer_index),
            most_positive_activation_records=most_positive_activation_records,
            random_sample=random_sample,
            activation_count=neuron_activation_counts[neuron_idx],
        )

        neuron_records.append(neuron_record)

    # now even add the neurons that were not activated
    for neuron_idx in set(neuron_activation_counts.keys()).difference(
        set(neuron_activations.keys())
    ):
        # sanity check
        assert neuron_activation_counts[neuron_idx] == 0
        # Create a NeuronRecord object
        neuron_record = NeuronRecord(
            neuron_id=NeuronId(neuron_index=neuron_idx, layer_index=layer_index),
            most_positive_activation_records=[],
            random_sample=[],
            activation_count=neuron_activation_counts[neuron_idx],
        )

        neuron_records.append(neuron_record)

    neuron_records = NeuronRecords(records=neuron_records)
    return neuron_records
