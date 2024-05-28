import wandb
import torch
from typing import Type, Dict


def wandb_log(data):
    try:
        wandb.log(data)
    except wandb.errors.Error:
        pass


def wandb_config_log(data):
    try:
        wandb.config.update(data)
    except wandb.errors.Error:
        pass


def log_dict(data, config=False):
    if not config:
        wandb_log(data)
    else:
        wandb_config_log(data)
    for k, v in data.items():
        print(f"{k}: {v}")


# utils for loading and saving modles to disk


def get_extended_state_dict(module: torch.nn.Module, **extra_config) -> Dict[str, Dict]:
    """
    Get the extended state dict of a module
    """
    state_dict = module.state_dict()
    config_dict = {
        k: v
        for k, v in module.__dict__.items()
        if not k.startswith("_") and k != "training"
    }
    return {"state_dict": state_dict, "config": {**config_dict, **extra_config}}


def load_from_extended_state_dict(
    module_cls: Type[torch.nn.Module],
    state_dict: Dict[str, Dict],
    strict: bool = False,
    **config_overrides,
):
    """
    Load a model from an extended state dict
    """

    # check if state_dict is extended
    if "config" not in state_dict:
        # we assume this will be a normal state dict
        # allow for config overrides
        module = module_cls(**config_overrides)
        module.load_state_dict(state_dict, strict=strict)
        return module

    config = {**state_dict["config"], **config_overrides}

    module = module_cls(**config)
    info = module.load_state_dict(state_dict["state_dict"], strict=strict)

    # log
    if info.missing_keys:
        print(f"Missing keys: {info.missing_keys}")
    if info.unexpected_keys:
        print(f"Unexpected keys: {info.unexpected_keys}")

    return module
