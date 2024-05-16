import torch
from torch import nn
from torch.nn import functional as F


class TiedSparseAutoEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.M = nn.Parameter(
            torch.randn(in_features, hidden_dim),
            requires_grad=True,
        )
        self.activation = nn.ReLU()

    def encode(self, x):
        self._normalize_M()

        c = self.activation(x @ self.M)
        return c

    def decode(self, c):
        self._normalize_M()

        x_hat = c @ self.M.T
        return x_hat

    def _normalize_M(self):
        # row wise normalize M
        self.M.data = nn.functional.normalize(self.M, p=2, dim=0)

    def forward(self, x):
        c = self.encode(x)
        x_hat = self.decode(c)
        return x_hat, c

    def losses(self, x, c, x_hat, l1_coeff=1e-3):
        reconstruction_loss = F.mse_loss(x, x_hat)
        sparsity_loss = l1_coeff * torch.linalg.norm(c, ord=1, dim=-1).mean()
        return reconstruction_loss, sparsity_loss

    def init_weights(self, strategy="xavier", data_loader=None):
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
