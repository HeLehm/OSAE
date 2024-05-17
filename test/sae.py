import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.sae import SparseAutoEncoder

import unittest
import torch
import torch.nn as nn


class TestSparseAutoEncoder(unittest.TestCase):
    def setUp(self):
        self.in_features = 16
        self.hidden_dim = 32
        self.batch_size = 32

    def test_normalization_of_D_tied(self):
        model = SparseAutoEncoder(self.in_features, self.hidden_dim, tied=True)
        model.init_weights_M_(strategy="xavier")

        self._test_D_norm(model)

    def test_normalization_of_D_untied(self):
        model = SparseAutoEncoder(self.in_features, self.hidden_dim, tied=False)
        model.init_weights_D_(strategy="xavier")

        self._test_D_norm(model)

    def _test_D_norm(self, model: SparseAutoEncoder):
        prev_D = model.D.clone()
        model.normalize_D()
        # Get the decoder matrix D
        D = model.D

        # assert that D has changed
        self.assertFalse(torch.allclose(D, prev_D), "D was not updated.")

        # Calculate the norm of each column
        column_norms = D.norm(p=2, dim=0)

        # Check if all column norms are close to 1
        self.assertTrue(
            torch.allclose(column_norms, torch.ones_like(column_norms), atol=1e-6),
            f"Column norms of D are not normalized: {column_norms}",
        )

    def test_init_weights_D_tied(self):
        model = SparseAutoEncoder(self.in_features, self.hidden_dim, tied=True)
        torch.manual_seed(42)
        old_M = model.M.clone()
        model.init_weights_D_(strategy="orthogonal")
        # make sure M even changed
        self.assertFalse(torch.allclose(model.M, old_M), "M was not updated.")

        # M.T should be initialized orthognal
        torch.manual_seed(42)
        test_D = torch.empty_like(model.M.T)
        nn.init.orthogonal_(test_D)

        self.assertTrue(
            torch.allclose(model.D, test_D), "M.T was not initialized orthogonal"
        )


if __name__ == "__main__":
    unittest.main()
