
from torch import nn

class HookedModel(nn.Module):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self._hook_temp_storage = {}

    def register_forward_hook_for_name(self, layer_name):
        #layer = getattr(self.model, name)
        layer = self._get_layer_by_name(layer_name)
        layer.register_forward_hook(self._get_activation_hook(layer_name))


    def forward(self, *args, **kwargs):
        self._hook_temp_storage = {}
        res = self.model(*args, **kwargs)
        return res, self._copy_reset_hook_results()
    
    def _get_activation_hook(self, name):
        """
        Get the hook function
        Usage:
        ```
        self.clip_model.visual.transformer.resblocks[0]\
            .register_forward_hook(self._get_activation_hook('visual_transformer_resblocks0'))
        ```
        """

        def hook(model, input, output):
            self._hook_temp_storage[name] = output

        return hook

    def _copy_reset_hook_results(self):
        """
        Get the hook results and reset the temp storage
        """
        hook_res = {k: v for k, v in self._hook_temp_storage.items()}
        self._hook_temp_storage = {}
        return hook_res
    
    def _get_layer_by_name(self, name):
        """
        Traverse the model to get the layer specified by the dot notation name
        """
        attrs = name.split('.')
        layer = self.model
        for attr in attrs:
            layer = getattr(layer, attr)
        return layer
    
    def __getattr__(self, name):
        """
        Redirect attribute access to the underlying model if not found in this class
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)