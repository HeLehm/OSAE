import os
import sys
import unittest
import torch
import tempfile

# append src path to sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.orthogonal_sae import OrthogonalSAE
from src.sae import SparseAutoEncoder


class TestClipSegSaveLoad(unittest.TestCase):
    def test_save_load_OSAE(self):
        self._test_save_load(
            OrthogonalSAE,
            in_features=10,
            hidden_dim=20,
            bias=False,
            allow_shear=False,
            tied=False,
        )

    def test_save_load_SAE(self):
        self._test_save_load(
            SparseAutoEncoder,
            in_features=10,
            hidden_dim=20,
            bias=False,
            tied=False,
        )

    @torch.no_grad()
    def _test_save_load(self, model_cls, **decoder_kwargs):
        model = model_cls(**decoder_kwargs)

        model.eval()

        random_input = torch.randn(10, model.in_features)

        random_output, random_act = model(random_input)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.pth")
            model.save(model_path)

            del model

            model = model_cls.load(model_path)
            model.eval()

        new_output, new_act = model(random_input)

        self.assertTrue(
            torch.allclose(random_output, new_output),
            f"Max diff {torch.max(torch.abs(random_output - new_output))}, average diff {torch.mean(torch.abs(random_output - new_output))}",
        )
        self.assertTrue(
            torch.allclose(random_act, new_act),
            f"Max diff {torch.max(torch.abs(random_act - new_act))}, average diff {torch.mean(torch.abs(random_act - new_act))}",
        )


if __name__ == "__main__":
    unittest.main()
