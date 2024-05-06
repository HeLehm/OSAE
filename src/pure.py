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

def gradient_activation_attribution(model, neuron_idx, input_ids, layer=-1):
    """
    Computes the Gradient × Activation attribution for a given neuron in a given layer.
    
    Args:
    - model: The HookedTransformer model.
    - layer (L or l in PURE paper): The layer index. Can be negative to index from the last layer.
    - neuron_idx (k in PURE paper): The neuron index in the given layer.
    - input_ids: The input IDs for the forward pass.
    
    Returns:
    - attributions: Gradient × Activation attributions for the previous layer.
        with shape (num_tokens, num neurons in prev layer).
        (R^{L-1} in the PURE paper), where the 2nd dimension in i in the paper (R^{L-1}_i).
    """
    if layer < 0:
        num_layers = model.cfg.n_layers
        layer = num_layers + layer
    
    model.eval()
    _, cache = get_mlp_cache(model, input_ids)
    
    # Get the activation A^L_k of the specified neuron
    lay_min_1_key = f'blocks.{layer - 1}.hook_mlp_out'
    lay_key = f'blocks.{layer}.hook_mlp_out'

    activation_L_minus_1 = cache[lay_min_1_key]
    activation_L = cache[lay_key]

    # Compute the gradient of the output neuron w.r.t. the lower-level activations
    grad = torch.autograd.grad(
        activation_L[0, 0, neuron_idx],
        activation_L_minus_1,
        retain_graph=True
    )[0]
    grad = grad[0, :, :]  # Extract the gradient of the relevant neuron

    # Compute Gradient × Activation
    attributions = activation_L_minus_1[0, :, :] * grad

    return attributions