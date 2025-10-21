
import unittest
import torch
from cbert.model import create_cbert_model
from transformers import BertForMaskedLM, BertConfig

class TestModel(unittest.TestCase):

    def test_create_cbert_model_config(self):
        vocab_size = 1000
        # Create a dummy config to pass to the model creation function
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            type_vocab_size=2,
        )
        model = create_cbert_model(config)

        # Check if the returned object is a BertForMaskedLM model
        self.assertIsInstance(model, BertForMaskedLM)

        # Check if the vocab size is set correctly
        self.assertEqual(model.config.vocab_size, vocab_size)

        # Check other BERT-base parameters
        self.assertEqual(model.config.hidden_size, 768)
        self.assertEqual(model.config.num_hidden_layers, 12)
        self.assertEqual(model.config.num_attention_heads, 12)
        self.assertEqual(model.config.max_position_embeddings, 512)

    def test_create_cbert_model_output_shape(self):
        vocab_size = 100
        max_seq_len = 50
        # Create a dummy config to pass to the model creation function
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=64, # Smaller for faster test
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=max_seq_len,
            type_vocab_size=2,
        )
        model = create_cbert_model(config)
        model.eval() # Set to eval mode for inference

        # Create dummy input
        input_ids = torch.randint(0, vocab_size, (1, max_seq_len)) # Batch size 1
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        # Check logits shape: (batch_size, sequence_length, vocab_size)
        self.assertEqual(outputs.logits.shape, (1, max_seq_len, vocab_size))

    def test_create_cbert_model_num_parameters(self):
        vocab_size = 128 # A typical small vocab size for char-level
        # Create a dummy config to pass to the model creation function
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            type_vocab_size=2,
        )
        model = create_cbert_model(config)
        num_params = model.num_parameters()
        
        # Assert that the number of parameters is within a reasonable range for a BERT-base like model
        # This value will depend on hidden_size, num_hidden_layers etc.
        # For hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072
        # and vocab_size=128, the number of parameters should be around 109M (BERT-base) + embedding for vocab_size
        # A rough check for a small vocab size
        expected_min_params = 100_000_000 # Lower bound for a small BERT-base
        expected_max_params = 120_000_000 # Upper bound
        self.assertGreater(num_params, expected_min_params)
        self.assertLess(num_params, expected_max_params)

if __name__ == '__main__':
    unittest.main()
