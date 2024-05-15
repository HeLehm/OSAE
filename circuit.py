import torch
from src.pure import gradient_activation_attribution
from src.model import get_model_and_tokenizer

model, tokenizer = get_model_and_tokenizer()


example_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "apple is a fruit.",
]
input_ids = tokenizer(
    example_sentences, return_tensors="pt", padding=True, truncation=True
)["input_ids"]
print(input_ids)

neuron_idx = 0
attributions = gradient_activation_attribution(
    model, input_ids, neuron_idx, layer=5, token_idx=1
)
print(attributions.shape)

# make sure the 2 bacth elements are differnet
assert not torch.allclose(attributions[0], attributions[1])
