from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F


class SparseAutoEncoder(nn.Module):
    def __init__(
        self, in_features, hidden_dim, tied=True, bias=True, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.M = nn.Parameter(
            torch.randn(in_features, hidden_dim),
            requires_grad=True,
        )
        self.activation = nn.ReLU()

        if tied:
            self.register_parameter("_D", None)
        else:
            self._D = nn.Parameter(
                torch.randn(hidden_dim, in_features),
                requires_grad=True,
            )

        if bias:
            self.bias = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)
        else:
            self.register_parameter("bias", None)

        self.init_weights_bias_()

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
        if self.bias is not None:
            c += self.bias

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

    def losses(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        x_hat: torch.Tensor,
        l1_coeff: float = 1e-3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the reconstruction and sparsity losses.
        """
        reconstruction_loss = F.mse_loss(x, x_hat)
        sparsity_loss = l1_coeff * torch.linalg.norm(c, ord=1, dim=-1).mean()
        return reconstruction_loss, sparsity_loss

    def init_weights_D_(self, strategy: str):
        self._init_weight_(self.D, strategy)

    def init_weights_M_(self, strategy: str):
        self._init_weight_(self.M, strategy)

    def init_weights_bias_(self):
        if self.bias is None:
            return
        nn.init.normal_(self.bias)

    def _init_weight_(self, weight, strategy):
        if strategy == "xavier":
            nn.init.xavier_normal_(weight)
        elif strategy == "orthogonal":
            nn.init.orthogonal_(weight)
        else:
            raise ValueError("Invalid strategy")


if __name__ == "__main__":
    model = SparseAutoEncoder(128, 512)
    model.init_weights(strategy="orthogonal")  # Ensure weights are initialized

    x = torch.randn(32, 128)
    x_hat, c = model(x)
    reconstruction_loss, sparsity_loss = model.losses(x, c, x_hat)
    print(
        f"Reconstruction Loss: {reconstruction_loss.item()}, Sparsity Loss: {sparsity_loss.item()}"
    )
