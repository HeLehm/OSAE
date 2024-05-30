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
from typing import Optional, Union, List, Dict

from neuron_explainer.activations.activations import NeuronRecord as _NeuronRecord

from neuron_explainer.activations.activations import (
    ActivationRecord,
    NeuronId,
    ActivationRecordSliceParams,
)
from neuron_explainer.explanations.explainer import TokenActivationPairExplainer
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.calibrated_simulator import (
    UncalibratedNeuronSimulator,
)
from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator
from neuron_explainer.activations.activation_records import calculate_max_activation
from neuron_explainer.explanations.scoring import simulate_and_score

from .act_dataset import ActivationDataset


from neuron_explainer.fast_dataclasses import (
    FastDataclass,
    register_dataclass,
    dumps,
    loads,
)
from dataclasses import dataclass, field


@register_dataclass
@dataclass
class NeuronRecord(_NeuronRecord):
    activation_count: int = 0


@register_dataclass
@dataclass
class NeuronRecords(FastDataclass):
    """dataclass that stores multiple neuron records"""

    records: List[NeuronRecord] = field(default_factory=list)

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(dumps(self).decode("utf-8"))

    @staticmethod
    def load(path: str):
        with open(path, "r") as f:
            return loads(f.read())


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
    neuron_activation_counts = defaultdict(int)

    # Iterate over activation records
    for i, (neuron_idx, activation_record) in enumerate(
        activation_record_iter(
            activation_ds, sae, idx_filter, tqdm_desc="Generating Neuron Records"
        )
    ):
        # if a neuron is not activated, we do not want to store it
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

    neuron_records = NeuronRecords(records=neuron_records)
    return neuron_records


async def interpret(
    neuron_records: NeuronRecords,
    top_k: int = 20,
    eval_num_features: int = 150,
    explainer_model_name: str = "gpt-4",
    simulator_model_name: str = "text-davinci-003",
) -> List[Dict[str, Union[int, str, float]]]:
    """
    Interpret the features of the model using OpenAI's models
    """
    neuron_records = neuron_records.records
    assert (
        len(neuron_records) > eval_num_features
    ), "Number of features should be greater than eval_num_features"

    # initialize explainer
    explainer = TokenActivationPairExplainer(
        model_name=explainer_model_name,
        prompt_format=PromptFormat.HARMONY_V4,
        max_concurrent=1,
    )

    # sort by activation count
    neuron_records.sort(key=lambda x: x.activation_count, reverse=True)
    # drop all that are not activated at least top_k times
    neuron_records = [x for x in neuron_records if x.activation_count > top_k]

    assert len(neuron_records) > eval_num_features, "Not enough features to evaluate"

    # select records (take high frequency neurons & low frequency neurons)
    neuron_records_to_evaluate = (
        neuron_records[: eval_num_features // 2]
        + neuron_records[-eval_num_features // 2 :]
    )

    results = []

    for neuron_record in tqdm(neuron_records_to_evaluate, desc="Interpreting Features"):
        try:
            # info about which feature we are evaluating
            neuron_id: int = neuron_record.neuron_id.neuron_index
            activation_count: int = neuron_record.activation_count

            # Grab the activation records we'll need.
            slice_params = ActivationRecordSliceParams(n_examples_per_split=5)
            train_activation_records = neuron_record.train_activation_records(
                activation_record_slice_params=slice_params
            )
            valid_activation_records = neuron_record.valid_activation_records(
                activation_record_slice_params=slice_params
            )

            explanations = await explainer.generate_explanations(
                all_activation_records=train_activation_records,
                max_activation=calculate_max_activation(train_activation_records),
                num_samples=1,
            )

            if len(explanations) > 1:
                print(f"Multiple explanations for neuron {neuron_id}")
                print("Taking the first explanation")
                print(explanations)

            explanation = explanations[0]

            # initialize simulator,
            simulator = UncalibratedNeuronSimulator(
                ExplanationNeuronSimulator(
                    simulator_model_name,
                    explanation,
                    max_concurrent=1,
                    prompt_format=PromptFormat.INSTRUCTION_FOLLOWING,
                )
            )

            scored_simulation = await simulate_and_score(
                simulator, valid_activation_records
            )
            score = scored_simulation.get_preferred_score()

            record = {
                "neuron_id": neuron_id,
                "explanation": explanation,
                "score": score,
                "activation_count": activation_count,
            }

            results.append(record)

        except Exception as e:
            print(f"Error for neuron {neuron_id}: {e}")
            continue

    return results
