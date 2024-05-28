import os
import torch
import numpy as np
from torch.utils.data import Dataset
from datasets import Dataset as hfDataset
from transformers import AutoModel
from tqdm import tqdm

from .custom_hooked import HookedModel
from .paths import get_embeddings_cache_dir

from typing import Union, Tuple, List
import json


class ActivationDataset(Dataset):
    def __init__(
        self,
        layername: str,
        model: Union[AutoModel, HookedModel],
        text_dataset: hfDataset,
        cache_root_dir=None,
        flatten_sequence=True,
        max_length=128,
        **kwargs,
    ) -> None:
        super().__init__()

        if cache_root_dir is None:
            cache_root_dir = get_embeddings_cache_dir()
        self.cache_root_dir = cache_root_dir

        self.flatten_sequence = flatten_sequence
        self.max_length = max_length
        self.layername = layername
        self.model_name = model.name_or_path if not isinstance(model, str) else model
        self.text_dataset_name = (
            text_dataset.info.dataset_name + "/" + text_dataset.split._name
            if not isinstance(text_dataset, str)
            else text_dataset
        )

        if not self.exists():
            self.generate(model, text_dataset, **kwargs)

        self.activation_cache_file_name = self.get_activations_cache_name()
        self.data_shape = self._get_data_shape(model, text_dataset)
        self.data = np.memmap(
            self.activation_cache_file_name,
            dtype="float32",
            mode="r",
            shape=self.data_shape,
        )
        self.corresponding_texts = self.load_corresponding_texts()

        assert self.data_shape[0] == len(self.corresponding_texts)
        assert (
            self.data[0].shape[0] == len(self.corresponding_texts[0]) == self.max_length
        )

        print("ActivationDataset with shape:", self.shape)

    @property
    def shape(self):
        if not self.flatten_sequence:
            return self.data_shape
        return (self.data_shape[0] * self.max_length, self.data_shape[2])

    @property
    def feature_dim(self):
        return self.data_shape[-1]

    def _make_model_hooked(self, model):
        if isinstance(model, str):
            model = AutoModel.from_pretrained(model)

        if not isinstance(model, HookedModel):
            model = HookedModel(model)

        model.register_forward_hook_for_name(self.layername)
        model.eval()

        return model

    @torch.no_grad()
    def generate(
        self,
        model,
        text_dataset: hfDataset,
        batch_size=8,
        num_proc=4,
        device=None,
    ):
        model = self._make_model_hooked(model)

        def tokenize_function(examples):
            val = model.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_offsets_mapping=True,
            )

            # add corresponding text for each token/ later activation

            val["corresponding_text"] = [
                [
                    examples["text"][test_i][start:end]
                    for start, end in val["offset_mapping"][test_i]
                ]
                for test_i in range(len(examples["text"]))
            ]

            del val["offset_mapping"]

            return val

        text_dataset = text_dataset.map(
            tokenize_function,
            batch_size=1000,
            batched=True,
            num_proc=num_proc,
        )

        text_dataset = text_dataset.remove_columns(["meta", "text"])
        text_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        if device is not None:
            model.to(device)

        device = next(model.parameters()).device

        dataloader = torch.utils.data.DataLoader(
            text_dataset, batch_size=batch_size, shuffle=False
        )

        first_batch = next(iter(dataloader))
        first_batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in first_batch.items()
        }
        _, cache = model(**first_batch)
        example_activation = cache[self.layername]
        if isinstance(example_activation, tuple):
            example_activation = example_activation[0]
        example_activation = example_activation.detach().cpu()
        activation_shape = example_activation.shape

        print("Generating dataset with activation shape:", activation_shape[1:])

        total_samples = len(text_dataset)
        memmap_shape = (total_samples,) + activation_shape[1:]
        activations = np.memmap(
            self.get_activations_cache_name(),
            dtype="float32",
            mode="w+",
            shape=memmap_shape,
        )
        # save corresponding text
        corresponding_text = text_dataset["corresponding_text"]
        with open(self.get_correponding_text_cache_name(), "w") as f:
            json.dump(corresponding_text, f)

        start_idx = 0
        for batch in tqdm(dataloader, desc="Generating Activations"):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            _, cache = model(**batch)
            layer_act = cache[self.layername]
            if isinstance(layer_act, tuple):
                layer_act = layer_act[0]
            layer_act = layer_act.detach().cpu().numpy()
            end_idx = start_idx + layer_act.shape[0]
            activations[start_idx:end_idx] = layer_act
            start_idx = end_idx

        activations.flush()

    def get_cache_dir(self):
        p = os.path.join(
            self.cache_root_dir,
            self.model_name,
            self.text_dataset_name,
            str(self.max_length),
        )
        os.makedirs(p, exist_ok=True)
        return p

    def get_activations_cache_name(self):
        p = os.path.join(self.get_cache_dir(), self.layername, "activations.npy")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def get_correponding_text_cache_name(self):
        return self.get_activations_cache_name().replace(
            "activations.npy", "corresponding_text.json"
        )

    @torch.no_grad()
    def _get_data_shape(self, model, text_dataset):
        # pass dummy input to model
        model = self._make_model_hooked(model)
        _, cache = model(**model.dummy_inputs)
        example_activation = cache[self.layername]
        if isinstance(example_activation, tuple):
            example_activation = example_activation[0]
        example_activation = example_activation.detach().cpu()
        return (len(text_dataset), self.max_length, example_activation.shape[-1])

    def exists(self):
        return os.path.exists(self.get_activations_cache_name()) and os.path.exists(
            self.get_correponding_text_cache_name()
        )

    def load_corresponding_texts(self):
        with open(self.get_correponding_text_cache_name(), "r") as f:
            return json.load(f)

    def __len__(self):
        if self.flatten_sequence:
            return self.data.shape[0] * self.max_length
        return self.data_shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Union[str, List[str]]]:
        _idx = idx
        if self.flatten_sequence:
            _idx, _idx2 = divmod(idx, self.max_length)
            return torch.from_numpy(
                self.data[_idx, _idx2].copy()
            ), self.corresponding_texts[_idx][_idx2]
        return torch.from_numpy(self.data[_idx].copy()), self.corresponding_texts[_idx]
