from typing import Tuple
import torch
from torch import nn
from ..utils import get_extended_state_dict, load_from_extended_state_dict
from .utils import householder


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


class OrthogonalSAE(torch.nn.Module):
    """
    Orthogonal Sparse Autoencoder (OrthogonalSAE).

    This class implements an autoencoder with an additional focus on enforcing orthogonality
    in the decoder through the use of Householder transformations. It ensures that the
    learned representations maintain certain orthogonal properties.
    """

    def __init__(
        self,
        in_features,
        hidden_dim,
        bias=True,
        allow_shear=True,
        tied=False,
        **kwargs,
    ) -> None:
        # hidden dim should either be the same as in_features or twice as large
        if hidden_dim != in_features and hidden_dim != in_features * 2:
            raise ValueError(
                "The hidden dimension should either be the same as the input dimension or twice as large"
            )

        # store these so we can save and load the model
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.allow_shear = allow_shear
        self.tied = tied

        super().__init__()
        if not tied:
            self._M = torch.nn.Parameter(
                torch.randn(in_features, hidden_dim),
                requires_grad=True,
            )
        else:
            self.register_parameter("_M", None)

        if bias:
            self.bias_weight = torch.nn.Parameter(
                torch.randn(hidden_dim),
                requires_grad=True,
            )
        else:
            self.register_parameter("bias_weight", None)

        decoder_allowed_directions_value = (
            torch.cat(
                (
                    torch.eye(in_features, in_features),
                    torch.eye(in_features, in_features) * -1,
                ),
            )
            if hidden_dim == in_features * 2
            else torch.eye(hidden_dim, in_features)
        )

        # a orthonormal basis for the decoder
        # not trainable, so it will stay orthogonal
        # possible shapes:
        # - (in_features * 2, in_features)
        # - (in_features, in_features)
        self.decoder_allowed_directions = torch.nn.Parameter(
            decoder_allowed_directions_value,
            requires_grad=False,
        )

        # the refelction evctor that will be learned
        # this will be the direction te ctor [1,0,0,..] will be reflected to
        # the direction will be applied ti the allowed directions to get the decoder matrixs
        self.decoder_reflection_vec = torch.nn.Parameter(
            torch.randn(hidden_dim),
            requires_grad=True,
        )

        if not allow_shear:
            self.register_parameter("shear_param", None)
        else:
            num_elements_in_triu = (
                torch.triu_indices(hidden_dim, hidden_dim, 1).numel() // 2
            )
            self.shear_param = torch.nn.Parameter(
                torch.zeros(num_elements_in_triu),
                requires_grad=True,
            )

        # the activation function after the encoder
        self.activation = torch.nn.ReLU()

        # just for faster decoder weight computation
        self._up_vector = torch.nn.Parameter(
            torch.zeros(hidden_dim),
            requires_grad=False,
        )
        self._up_vector.data[0] = 1

    @property
    def M(self) -> torch.Tensor:
        if self._M is not None:
            return self._M
        else:
            return self.D.T

    @property
    def D(self) -> torch.Tensor:
        """
        Compute the decoder weight matrix from the reflection vector.
        """
        u = self.decoder_reflection_vec / self.decoder_reflection_vec.norm()
        H = householder(u, self._up_vector)
        decoder_weight = H @ self.decoder_allowed_directions

        if self.shear_param is not None:
            shear = torch.eye(
                decoder_weight.shape[0],
                decoder_weight.shape[0],
                device=decoder_weight.device,
            )
            triu_indices = torch.triu_indices(
                decoder_weight.shape[0], decoder_weight.shape[0], 1
            )
            shear[triu_indices[0], triu_indices[1]] = self.shear_param
            decoder_weight = shear @ decoder_weight

        decoder_weight = decoder_weight / torch.linalg.norm(
            decoder_weight, ord=2, dim=0
        )
        return decoder_weight

    def encode(self, x):
        """
        Encode the input x using the encoder matrix M.
        x: (batch_size, in_features)
        return: (batch_size, hidden_dim)
        """
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
        x_hat = c @ self.D
        return x_hat

    def forward(self, x):
        c = self.encode(x)
        x_hat = self.decode(c)
        return x_hat, c

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
