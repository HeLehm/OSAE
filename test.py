from transformers import AutoModel
import torch
from src.backbone import get_backbone

model = get_backbone("EleutherAI/pythia-70m-deduped")
exmaple_text = ["This is a test", "This is another test"]
sample_tokens = model.tokenizer(
                exmaple_text,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            )

print(model)

# model = AutoModel.from_pretrained("EleutherAI/pythia-70m-deduped")

# for name, param in model.named_parameters():
#     print(name)

# print()

# from datasets import load_dataset

# ds = load_dataset("NeelNanda/pile-10k")


# from src.custom_hooked import HookedModel

# hooked_model = HookedModel(model)


# hooked_model.register_forward_hook_for_name("layers.5.mlp")

# output, cache = hooked_model(**hooked_model.model.dummy_inputs)

# output = output.last_hidden_state


# print("out", output.shape)
# for k,v in cache.items():
#     print(k, v.shape)

# assert torch.allclose(output, cache["final_layer_norm"])

