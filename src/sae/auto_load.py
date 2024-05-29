# util to auto load based on state dict
import torch
from ..utils import load_from_extended_state_dict


def load_sae(path):
    """
    Load the model from disk.
    """
    state_dict = torch.load(path)
    cls = state_dict["config"]["cls"]
    model = load_from_extended_state_dict(cls, state_dict, strict=False)
    return model
