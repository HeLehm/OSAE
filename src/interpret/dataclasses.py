from neuron_explainer.activations.activations import NeuronRecord as _NeuronRecord
from neuron_explainer.fast_dataclasses import (
    FastDataclass,
    register_dataclass,
    dumps,
    loads,
)
from dataclasses import dataclass, field
from typing import List


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
