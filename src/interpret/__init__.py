# make sure an openai api key is set
import os

if "OPENAI_API_KEY" not in os.environ:
    # warn the user and set the key to a fake
    print("OPENAI_API_KEY not set, setting to fake key")
    os.environ["OPENAI_API_KEY"] = "fake_key"


from .dataclass_utils import generate_neuron_records
from .dataclasses import NeuronRecords, NeuronRecord
from .interpret_functions import interpret_neuron_record, interpret_neuron_records

__all__ = [
    "generate_neuron_records",
    "NeuronRecords",
    "NeuronRecord",
    "interpret_neuron_record",
    "interpret_neuron_records",
]
