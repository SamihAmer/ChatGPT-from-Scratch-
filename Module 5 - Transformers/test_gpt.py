import os
import sys
import unittest

import torch


# Allow `import gpt` (and its sibling modules) from this folder even when
# running tests from the repository root.
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from gpt import GPTModel  # noqa: E402


class TestGPTModel(unittest.TestCase):
    def test_forward_shape(self):
        torch.manual_seed(0)
        vocab_size = 100
        model = GPTModel(d_model=32, n_heads=4, layers=2, vocab_size=vocab_size, max_seq_len=16)
        x = torch.randint(vocab_size, (3, 7), dtype=torch.long)
        y = model(x)
        self.assertEqual(tuple(y.shape), (3, 7, vocab_size))

    def test_token_embedding_uses_vocab_size(self):
        vocab_size = 123
        model = GPTModel(d_model=16, n_heads=4, layers=1, vocab_size=vocab_size, max_seq_len=8)
        self.assertEqual(model.token_embedding.weight.shape[0], vocab_size)

    def test_raises_when_sequence_too_long(self):
        model = GPTModel(d_model=16, n_heads=4, layers=1, vocab_size=50, max_seq_len=4)
        x = torch.randint(50, (2, 5), dtype=torch.long)
        with self.assertRaises(ValueError):
            _ = model(x)

    def test_causal_invariance(self):
        torch.manual_seed(0)
        vocab_size = 50
        model = GPTModel(d_model=32, n_heads=4, layers=2, vocab_size=vocab_size, max_seq_len=16)
        model.eval()  # disable dropout for a deterministic causal check

        B, S = 2, 8
        x1 = torch.randint(vocab_size, (B, S), dtype=torch.long)
        x2 = x1.clone()

        cutoff = 3
        # Change tokens strictly after the cutoff; outputs up to cutoff should match.
        x2[:, cutoff + 1 :] = torch.randint(vocab_size, (B, S - (cutoff + 1)), dtype=torch.long)

        with torch.no_grad():
            y1 = model(x1)
            y2 = model(x2)

        torch.testing.assert_close(
            y1[:, : cutoff + 1, :],
            y2[:, : cutoff + 1, :],
            rtol=0.0,
            atol=1e-6,
        )

    def test_backward_runs(self):
        torch.manual_seed(0)
        vocab_size = 80
        model = GPTModel(d_model=32, n_heads=4, layers=2, vocab_size=vocab_size, max_seq_len=16)
        model.eval()

        x = torch.randint(vocab_size, (2, 6), dtype=torch.long)
        logits = model(x)
        loss = logits.mean()
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in grads))


if __name__ == "__main__":
    unittest.main()

