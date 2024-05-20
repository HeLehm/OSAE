import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


from src.metrics import SlidingWindowDeadNeuronTracker

import unittest
import torch


class TestSlidingWindowDeadNeuronDetector(unittest.TestCase):
    def setUp(self):
        self.max_len = 5
        self.embedding_dim = 3
        self.detector = SlidingWindowDeadNeuronTracker(self.max_len, self.embedding_dim)

    def test_initial_state(self):
        # if the das has not been filed yet, we hanlde it as if all neurons were active juts before the initialization
        steps_since_active = self.detector.get_inactive_period()
        # i.e. all neurson have ben active 'in the last batch'
        expected = torch.zeros(self.embedding_dim)
        self.assertTrue(
            torch.equal(steps_since_active, expected),
            f"steps_since_active: {steps_since_active}",
        )

    def test_on_batch_simple(self):
        for i in range(self.max_len):
            c = torch.tensor([[0.0, 1.0, 0.0]])
            inactive_period = self.detector.on_batch(c)
            # assert that no value is 1 yet
            # (that should only happen after max_len steps)
            if i < self.max_len - 1:
                self.assertTrue(
                    torch.all(inactive_period < 1),
                    f"steps_since_active: {inactive_period}",
                )

        inactive_period = self.detector.get_inactive_period()
        excepted = torch.tensor([1.0, 0.0, 1.0])
        self.assertTrue(
            torch.allclose(inactive_period, excepted),
            f"inactive_period: {inactive_period}",
        )

    def test_on_batch_comples(self):
        # like test_on_batch_simple but a neuron activates after 3 steps
        for i in range(self.max_len):
            c = torch.tensor([[0.0, 1.0, 0.0]])
            if i == 2:
                c = torch.tensor([[1.0, 0.0, 0.0]])
            inactive_period = self.detector.on_batch(c)
            # assert that no value is 1 yet
            # (that should only happen after max_len steps)
            if i < self.max_len - 1:
                self.assertTrue(
                    torch.all(inactive_period < 1),
                    f"inactive_period: {inactive_period}",
                )

        inactive_period = self.detector.get_inactive_period()
        excepted = torch.tensor([0.4, 0.0, 1.0])
        self.assertTrue(
            torch.allclose(inactive_period, excepted),
            f"inactive_period: {inactive_period}, expected: {excepted}",
        )


if __name__ == "__main__":
    unittest.main()
