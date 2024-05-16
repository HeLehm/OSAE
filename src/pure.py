import torch
from transformer_lens import HookedTransformer


def get_mlp_cache(model: HookedTransformer, input_ids: torch.Tensor, **kwargs):
    """
    Retrieves MLP outputs from all layers of the HookedTransformer during a forward pass.

    Args:
        model (HookedTransformer): The transformer model.
        input_ids (torch.Tensor): The input IDs for the forward pass.
        kwargs: Additional arguments for the model.

    Returns:
        tuple: The model output and a dictionary containing the MLP activations.
    """

    def filter_mlp_only(name):
        return "mlp" in name

    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act

    model.set_use_hook_mlp_in(True)
    model.reset_hooks()
    model.add_hook(filter_mlp_only, forward_cache_hook)
    output = model(input_ids, **kwargs)

    return output, cache


def gradient_activation_attribution(
    model: HookedTransformer,
    input_ids: torch.Tensor,
    neuron_idx: int,
    layer: int = -1,
    token_idx: int = 0,
):
    """
    Computes the Gradient x Activation attribution for a given neuron in a given layer.

    Args:
        model (HookedTransformer): The HookedTransformer model.
        input_ids (torch.Tensor): The input IDs for the forward pass.
        neuron_idx (int): The neuron index in the given layer.
        layer (int, optional): The layer index. Can be negative to index from the last layer. Defaults to -1.
        token_idx (int, optional): The token index for which the attribution is computed. Defaults to 0.

    Returns:
        torch.Tensor: Gradient x Activation attributions with shape (batch_size, num_tokens, num_neurons).
    """
    num_layers = model.cfg.n_layers
    layer = (num_layers + layer) if layer < 0 else layer

    # Check for valid layer index
    if not 0 <= layer < num_layers:
        raise ValueError(
            f"Invalid layer index {layer}. Must be between 0 and {num_layers - 1}."
        )

    _, cache = get_mlp_cache(model, input_ids)

    lay_min_1_key = f"blocks.{layer - 1}.hook_mlp_out" if layer > 0 else None
    lay_key = f"blocks.{layer}.hook_mlp_out"

    activation_L_minus_1 = cache[lay_min_1_key] if lay_min_1_key else None
    activation_L = cache[lay_key]

    if activation_L_minus_1 is None or activation_L is None:
        raise ValueError(
            f"Invalid cache. Layer {layer} or {layer - 1} activations not found."
        )

    batch_size, num_tokens, num_neurons = activation_L_minus_1.shape
    if not 0 <= neuron_idx < num_neurons:
        raise ValueError(
            f"Invalid neuron index {neuron_idx}. Must be between 0 and {num_neurons - 1}."
        )
    if not 0 <= token_idx < num_tokens:
        raise ValueError(
            f"Invalid token index {token_idx}. Must be between 0 and {num_tokens - 1}."
        )

    activation_L = activation_L[:, token_idx, neuron_idx]
    grads = torch.autograd.grad(
        activation_L.sum(), activation_L_minus_1, retain_graph=True
    )[0]
    attributions = activation_L_minus_1 * grads

    return attributions
