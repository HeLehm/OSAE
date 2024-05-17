import os
import sys

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
)

from src.sae import SparseAutoEncoder

import unittest
import torch

class TestSparseAutoEncoder(unittest.TestCase):
    def setUp(self):
        self.in_features = 128
        self.hidden_dim = 512
        self.batch_size = 32

    def test_normalization_of_D_tied(self):
        model = SparseAutoEncoder(self.in_features, self.hidden_dim, tied=True)
        model.init_weights(strategy="xavier")

        self._test_D_norm(model)

    def test_normalization_of_D_untied(self):
        model = SparseAutoEncoder(self.in_features, self.hidden_dim, tied=False)
        model.init_weights(strategy="xavier")

        self._test_D_norm(model)
        
    def _test_D_norm(self, model : SparseAutoEncoder):
        prev_D = model.D.clone()
        model.normalize_D()
        # Get the decoder matrix D
        D = model.D

        # assert that D has changed
        self.assertFalse(torch.allclose(D, prev_D), "D was not updated.")

        # Calculate the norm of each column
        column_norms = D.norm(p=2, dim=0)

        # Check if all column norms are close to 1
        self.assertTrue(torch.allclose(column_norms, torch.ones_like(column_norms), atol=1e-6),
                        f"Column norms of D are not normalized: {column_norms}")

if __name__ == "__main__":
    unittest.main()

