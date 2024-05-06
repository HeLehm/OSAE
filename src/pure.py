import torch
from transformer_lens import HookedTransformer

def get_mlp_cache(model: HookedTransformer, input_ids: torch.Tensor, **kwargs):
    filter_mlp_only = lambda name: "mlp" in name

    model.set_use_hook_mlp_in(True)
    model.reset_hooks()
    cache = {}

    def forward_cache_hook(act, hook):
        # Do not detach here
        cache[hook.name] = act

    model.add_hook(filter_mlp_only, forward_cache_hook)
    output = model(input_ids, **kwargs)
    return output, cache

def gradient_activation_attribution(model, layer, neuron_idx, input_ids):
    """
    Computes the Gradient × Activation attribution for a given neuron in a given layer.
    
    Args:
    - model: The HookedTransformer model.
    - layer: The layer index.
    - neuron_idx: The neuron index in the given layer.
    - input_ids: The input IDs for the forward pass.
    
    Returns:
    - attributions: Gradient × Activation attributions for the previous layer.
    """
    model.eval()
    _, cache = get_mlp_cache(model, input_ids)
    
    # Get the activation A^L_k of the specified neuron
    lay_min_1_key = f'blocks.{layer - 1}.hook_mlp_out'
    lay_key = f'blocks.{layer}.hook_mlp_out'

    activation_L_minus_1_full = cache[lay_min_1_key]
    activation_L_full = cache[lay_key]

    activation_L_minus_1 = activation_L_minus_1_full[0, 0, :]
    activation_L_k = activation_L_full[0, 0, neuron_idx]

    # Compute the gradient of the output neuron w.r.t. the lower-level activations
    grad = torch.autograd.grad(activation_L_k, activation_L_minus_1_full, retain_graph=True)[0]
    grad = grad[0, 0, :]  # Extract the gradient of the relevant neuron

    # Compute Gradient × Activation
    attributions = activation_L_minus_1 * grad

    return attributions

# Example usage
# Initialize your model, inputs, and call the function
# For example:
# model = HookedTransformer(...) 
# input_ids = torch.tensor([...])  
# gradient_activation_attribution(model, 5, 1, input_ids)
