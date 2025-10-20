import unittest
import torch
from cbert.cli.evaluate import calculate_accuracy, calculate_perplexity

class TestEvaluationMetrics(unittest.TestCase):

    def test_calculate_accuracy(self):
        logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2]], [[0.3, 0.7], [0.6, 0.4]]]) # (batch, seq_len, vocab)
        labels = torch.tensor([[1, 0], [0, 1]]) # (batch, seq_len)
        accuracy = calculate_accuracy(logits, labels)
        self.assertAlmostEqual(accuracy, 0.75)

    def test_calculate_perplexity(self):
        loss = torch.tensor(2.3)
        perplexity = calculate_perplexity(loss)
        self.assertAlmostEqual(perplexity, 9.974, places=3)

if __name__ == '__main__':
    unittest.main()
