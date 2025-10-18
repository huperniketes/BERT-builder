import unittest
import torch
from src.cli import evaluate # This will fail until the file is created

class TestEvaluationMetrics(unittest.TestCase):

    def test_accuracy_calculation(self):
        # This will fail until the function is implemented
        logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2]], [[0.3, 0.7], [0.6, 0.4]]]) # (batch, seq_len, vocab)
        labels = torch.tensor([[1, 0], [0, 1]]) # (batch, seq_len)
        accuracy = evaluate.calculate_accuracy(logits, labels)
        self.assertAlmostEqual(accuracy, 0.5)

    def test_perplexity_calculation(self):
        # This will fail until the function is implemented
        logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2]], [[0.3, 0.7], [0.6, 0.4]]]) # (batch, seq_len, vocab)
        labels = torch.tensor([[1, 0], [0, 1]]) # (batch, seq_len)
        # Dummy cross-entropy loss
        loss = 1.2
        perplexity = evaluate.calculate_perplexity(loss)
        self.assertAlmostEqual(perplexity, 3.32, places=2)

if __name__ == '__main__':
    unittest.main()
