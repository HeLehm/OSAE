import os
import torch
from typing import Union
from torch.utils.data import Dataset
from datasets import Dataset as hfDataset
from transformers import AutoModel
from tqdm import tqdm

from .custom_hooked import HookedModel
from .paths import get_embeddings_cache_dir

class ActivationDataset(Dataset):

    def __init__(
            self,
            layername : str,
            model : Union[AutoModel, HookedModel, str],
            text_dataset : Union[hfDataset, str],
            cache_root_dir=None,
            **kwargs
    ) -> None:
        super().__init__()

        if cache_root_dir is None:
            cache_root_dir = get_embeddings_cache_dir()
        self.cache_root_dir = cache_root_dir
        
        # hf model or HookedModel
        self.layername = layername
        self.model_name = model
        self.text_dataset_name = text_dataset

        # fix the names that are wrong
        if not isinstance(model, str):
            self.model_name = model.name_or_path
        if not isinstance(text_dataset, str):
            self.text_dataset_name = text_dataset.info.dataset_name + "/" + text_dataset.split._name

        if not self.exists():
            self.generate(model, text_dataset, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        model,
        text_dataset : hfDataset,
        batch_size=8,
        num_proc=4,
        max_length=256,
        device=None
    ):
        if isinstance(model, str):
            model = AutoModel.from_pretrained(model)
        
        if isinstance(text_dataset, str):
            raise ValueError("text_dataset must be a dataset object")

        if not isinstance(model, HookedModel):
            model = HookedModel(model)

        model.register_forward_hook_for_name(self.layername)
        model.eval()

        def tokenize_function(examples):
            val = model.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            return val
        # map with tokenizer
        text_dataset = text_dataset.map(
                tokenize_function,
                batch_size=1000,
                batched=True,
                num_proc=None,
            )
        # drop the columns that are not needed
        text_dataset = text_dataset.remove_columns(["text", "meta"])
        text_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

        if device is not None:
            model.to(device)

        device = next(model.parameters()).device

        dataloader = torch.utils.data.DataLoader(
            text_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        activations = []
        for batch in tqdm(dataloader, desc="Generating Activations"):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            _, cache = model(**batch)
            layer_act = cache[self.layername].detach().cpu()
            activations.append(layer_act)
        
        activations = torch.cat(activations, dim=0)
        torch.save(activations, self.get_cache_file_name())


    def get_cache_dir(self):
        """
        Returns something like:
        ./data/embeddings_cache/pythia-70m-deduped/pile-10k
        """
        p = os.path.join(self.cache_root_dir, self.model_name, self.text_dataset_name)
        os.makedirs(p, exist_ok=True)
        return p
    
    def get_cache_file_name(self):
        return os.path.join(self.get_cache_dir(), self.layername + ".pt")
    
    def exists(self):
        return os.path.exists(self.get_cache_file_name())