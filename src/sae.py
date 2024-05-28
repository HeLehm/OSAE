from typing import Tuple
import torch
from torch import nn
from .utils import get_extended_state_dict, load_from_extended_state_dict


class SparseAutoEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim, tied=True, bias=True, **kwargs) -> None:
        super().__init__()
        self.M = nn.Parameter(
            torch.randn(in_features, hidden_dim),
            requires_grad=True,
        )
        self.activation = nn.ReLU()

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.tied = tied
        self.bias = bias

        if tied:
            self.register_parameter("_D", None)
        else:
            self._D = nn.Parameter(
                torch.randn(hidden_dim, in_features),
                requires_grad=True,
            )

        if bias:
            self.bias_weight = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)
        else:
            self.register_parameter("bias_weight", None)

        # Initialize weights
        self.init_weights_bias_()
        self.init_weights_D_()
        if not tied:
            self.init_weights_M_()

    @property
    def D(self) -> nn.Parameter:
        if self._D is None:
            return self.M.T
        return self._D

    def encode(self, x):
        """
        Encode the input x using the encoder matrix M.
        x: (batch_size, in_features)
        return: (batch_size, hidden_dim)
        """
        self.normalize_D()

        c = x @ self.M
        if self.bias_weight is not None:
            c += self.bias_weight

        c = self.activation(c)
        return c

    def decode(self, c):
        """
        Decode the code c using the decoder matrix M.T.
        c: (batch_size, hidden_dim)
        return: (batch_size, in_features)
        """
        self.normalize_D()

        x_hat = c @ self.D
        return x_hat

    def normalize_D(self):
        """
        Normalize the decoder matrix D.
        NOTE: if tied M.T is D.

        The decoder Matrix D is a J (hidden_dim) x h (in_features) matrix (no bias).
        D can be M.T if tied.
        D is normalized by column i.e. the learned features are normalized.
        """
        with torch.no_grad():
            self.D.data.div_(torch.linalg.norm(self.D, ord=2, dim=0) + 1e-8)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the autoencoder.
        x: (batch_size, in_features)
        return: (batch_size, in_features), (batch_size, hidden_dim)
        """
        c = self.encode(x)
        x_hat = self.decode(c)
        return x_hat, c

    def init_weights_D_(self, strategy: str = "orthogonal"):
        self._init_weight_(self.D, strategy)

    def init_weights_M_(self, strategy: str = "orthogonal"):
        self._init_weight_(self.M, strategy)

    def init_weights_bias_(self):
        if self.bias_weight is None:
            return
        nn.init.normal_(self.bias_weight)

    def _init_weight_(self, weight, strategy):
        if strategy == "xavier":
            nn.init.xavier_normal_(weight)
        elif strategy == "orthogonal":
            nn.init.orthogonal_(weight)
        else:
            raise ValueError("Invalid strategy")

    def save(self, path):
        """
        Save the model to disk.
        """
        torch.save(
            get_extended_state_dict(self, cls=self.__class__),
            path,
        )

    @classmethod
    def load(cls, path, **config_overrides):
        """
        Load the model from disk.
        """
        state_dict = torch.load(path)
        return load_from_extended_state_dict(cls, state_dict, **config_overrides)


if __name__ == "__main__":
    model = SparseAutoEncoder(128, 512)
    model.init_weights(strategy="orthogonal")  # Ensure weights are initialized

    x = torch.randn(32, 128)
    x_hat, c = model(x)
    reconstruction_loss, sparsity_loss = model.losses(x, c, x_hat)
    print(
        f"Reconstruction Loss: {reconstruction_loss.item()}, Sparsity Loss: {sparsity_loss.item()}"
    )
