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

def gradient_activation_attribution(model, input_ids, neuron_idx, layer=-1, token_idx=0):
    """
    Computes the Gradient x Activation attribution for a given neuron in a given layer.
    only works for mlp_output layers for now.
    
    Args:
    - model: The HookedTransformer model.
    - layer (L or l in PURE paper): The layer index. Can be negative to index from the last layer.
    - neuron_idx (k in PURE paper): The neuron index in the given layer.
    - input_ids: The input IDs for the forward pass.
    - token_idx: The token index for which the attribution is computed.
        Note: will still return attributions for all prior tokens, but based on the gradient of the specified token.
    
    Returns:
    - attributions: Gradient x Activation attributions for the previous layer.
        with shape (bacth_size, num_tokens, num_neurons).
        (R^{L-1} in the PURE paper), where the last dimension in i in the paper (R^{L-1}_i).
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

    batch_size = activation_L_minus_1.shape[0]

    grads = []

    for i in range(batch_size):
        # Compute the gradient of the output neuron w.r.t. the lower-level activations
        grad = torch.autograd.grad(
            activation_L[i, token_idx, neuron_idx],
            activation_L_minus_1,
            retain_graph=True
        )[0]
        grad = grad[i, :, :]  # Extract the gradient of the relevant neuron
        grads.append(grad)

    grads = torch.stack(grads, dim=0)

    # Compute Gradient x Activation
    attributions = activation_L_minus_1 * grads

    return attributions
