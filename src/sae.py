import torch
from torch import nn
from torch.nn import functional as F


class SparseAutoEncoder(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_dim,
            tied=True,
            bias=True, *args, **kwargs
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
        
    @property
    def D(self):
        if self._D is None:
            return self.M.T
        return self._D

    def encode(self, x):
        """
        Encode the input x using the encoder matrix M.
        x: (batch_size, in_features)
        return: (batch_size, hidden_dim)
        """
        self._normalize_D()

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
        self._normalize_D()

        x_hat = c @ self.D
        return x_hat

    def _normalize_D(self):
        """
        Normalize the decoder matrix D.
        NOTE: if tied M.T is D.

        The decoder Matrix D is a J (hidden_dim) x h (in_features) matrix (no bias).
        D can be M.T if tied.
        D is normalized by column i.e. the learned features are normalized.
        """
        self.D.data = F.normalize(self.D, p=2, dim=-1)

    def forward(self, x):
        """
        Forward pass of the autoencoder.
        x: (batch_size, in_features)
        return: (batch_size, in_features), (batch_size, hidden_dim)
        """
        c = self.encode(x)
        x_hat = self.decode(c)
        return x_hat, c

    def losses(self, x, c, x_hat, l1_coeff=1e-3):
        """
        Compute the reconstruction and sparsity losses.
        """
        reconstruction_loss = F.mse_loss(x, x_hat)
        sparsity_loss = l1_coeff * torch.linalg.norm(c, ord=1, dim=-1).mean()
        return reconstruction_loss, sparsity_loss

    def init_weights(self, strategy="xavier", data_loader=None):
        if self.bias is not None:
            nn.init.normal_(self.bias)
        if strategy == "xavier":
            nn.init.xavier_normal_(self.M)
        elif strategy == "orthogonal":
            nn.init.orthogonal_(self.M)
        else:
            raise ValueError("Invalid strategy")


if __name__ == "__main__":
    model = SparseAutoEncoder(128, 512)
    x = torch.randn(32, 128)
    c = model.encode(x)
    x_hat = model.decode(c)
    print(x_hat.shape)
    diff = x - x_hat
    print(diff.abs().mean())
