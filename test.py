from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("EleutherAI/pythia-70m-deduped")

for name, param in model.named_parameters():
    print(name)

print()


from src.custom_hooked import HookedModel

hooked_model = HookedModel(model)


hooked_model.register_forward_hook_for_name("final_layer_norm")

output, cache = hooked_model(**hooked_model.model.dummy_inputs)

output = output.last_hidden_state


print("out", output.shape)
for k,v in cache.items():
    print(k, v.shape)

assert torch.allclose(output, cache["final_layer_norm"])

