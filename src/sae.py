import torch
from torch import nn
from torch.nn import functional as F


class TiedSparseAutoEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim, bias=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.M = nn.Parameter(
            torch.randn(in_features, hidden_dim),
            requires_grad=True,
        )
        self.activation = nn.ReLU()

        if bias:
            self.bias = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)
        else:
            self.register_parameter("bias", None)

    def encode(self, x):
        """
        Encode the input x using the encoder matrix M.
        x: (batch_size, in_features)
        return: (batch_size, hidden_dim)
        """
        self._normalize_M()

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
        self._normalize_M()

        x_hat = c @ self.M.T
        return x_hat

    def _normalize_M(self):
        """
        Normalize the decoder matrix M.

        M.T is the decoder matrix, also called dictionary D,
        which is a h (in_features) x J (hidden_dim) decoder matrix (no bias).
        M.T (D) is normalized by column i.e. the learned features are normalized.
        """
        self.M.data = F.normalize(self.M, p=2, dim=0)

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
    model = TiedSparseAutoEncoder(128, 512)
    x = torch.randn(32, 128)
    c = model.encode(x)
    x_hat = model.decode(c)
    print(x_hat.shape)
    diff = x - x_hat
    print(diff.abs().mean())
