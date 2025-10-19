
import unittest
from cbert.model import create_cbert_model
from transformers import BertForMaskedLM

class TestModel(unittest.TestCase):

    def test_create_cbert_model(self):
        vocab_size = 1000
        model = create_cbert_model(vocab_size)

        # Check if the returned object is a BertForMaskedLM model
        self.assertIsInstance(model, BertForMaskedLM)

        # Check if the vocab size is set correctly
        self.assertEqual(model.config.vocab_size, vocab_size)

        # Check other BERT-base parameters
        self.assertEqual(model.config.hidden_size, 768)
        self.assertEqual(model.config.num_hidden_layers, 12)
        self.assertEqual(model.config.num_attention_heads, 12)

if __name__ == '__main__':
    unittest.main()
