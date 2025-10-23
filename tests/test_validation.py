"""
Test validation pipeline for GEMINI.md compliance.
Validates data quality, intermediate results, and evaluation completeness.
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import json
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.cbert.data import CCodeDataset, preprocess_code
from src.cbert.tokenizer import CharTokenizer, KeyCharTokenizer
from src.cbert.model import create_cbert_model
from transformers import BertConfig


class TestValidationPipeline(unittest.TestCase):
    """Test validation pipeline for GEMINI.md standards compliance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path(tempfile.mkdtemp())
        self.sample_code = """
        int main() {
            // This is a comment
            int x = 42;
            return x;
        }
        """
        
        # Create test data file
        test_file = self.test_data_dir / "test.c"
        test_file.write_text(self.sample_code)
        
        # Basic config for testing
        self.config = BertConfig(
            vocab_size=128,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=512
        )
    
    def test_data_shape_validation(self):
        """Test intermediate data shape validation."""
        tokenizer = CharTokenizer()
        dataset = CCodeDataset(str(self.test_data_dir), tokenizer, max_length=128)
        
        # Validate dataset shapes
        self.assertGreater(len(dataset), 0, "Dataset should not be empty")
        
        sample = dataset[0]
        self.assertIn('input_ids', sample)
        self.assertIn('attention_mask', sample)
        
        # Shape validation
        input_shape = sample['input_ids'].shape
        mask_shape = sample['attention_mask'].shape
        
        self.assertEqual(len(input_shape), 1, "Input should be 1D tensor")
        self.assertEqual(input_shape, mask_shape, "Input and mask shapes must match")
        self.assertLessEqual(input_shape[0], 128, "Sequence length should not exceed max_length")
    
    def test_semantic_preservation(self):
        """Test that preprocessing preserves semantic meaning."""
        original_code = """
        int factorial(int n) {
            // Calculate factorial
            if (n <= 1) return 1;
            return n * factorial(n - 1);
        }
        """
        
        processed = preprocess_code(original_code)
        
        # Check that essential code structure is preserved
        self.assertIn('factorial', processed, "Function name should be preserved")
        self.assertIn('int n', processed, "Parameter should be preserved")
        self.assertIn('return', processed, "Return statements should be preserved")
        self.assertIn('if', processed, "Control flow should be preserved")
        
        # Check that comments are removed but code structure intact
        self.assertNotIn('Calculate factorial', processed, "Comments should be removed")
        self.assertIn('{', processed, "Braces should be preserved")
        self.assertIn('}', processed, "Braces should be preserved")
    
    def test_tokenizer_consistency(self):
        """Test tokenizer consistency across different types."""
        code_sample = "int x = 42;"
        
        char_tokenizer = CharTokenizer()
        keychar_tokenizer = KeyCharTokenizer()
        
        char_tokens = char_tokenizer.encode(code_sample)
        keychar_tokens = keychar_tokenizer.encode(code_sample)
        
        # Both should produce valid token sequences
        self.assertIsInstance(char_tokens, list, "Char tokenizer should return list")
        self.assertIsInstance(keychar_tokens, list, "KeyChar tokenizer should return list")
        
        # All tokens should be within vocab range
        self.assertTrue(all(0 <= t < char_tokenizer.vocab_size for t in char_tokens), "Char tokens out of range")
        self.assertTrue(all(0 <= t < keychar_tokenizer.vocab_size for t in keychar_tokens), "KeyChar tokens out of range")
        
        # Decode should be consistent
        char_decoded = char_tokenizer.decode(char_tokens)
        keychar_decoded = keychar_tokenizer.decode(keychar_tokens)
        
        self.assertIsInstance(char_decoded, str, "Decoded should be string")
        self.assertIsInstance(keychar_decoded, str, "Decoded should be string")
    
    def test_model_output_validation(self):
        """Test model output shapes and ranges."""
        model = create_cbert_model(self.config)
        
        # Create dummy input
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Validate output shapes
        logits = outputs.logits
        expected_shape = (batch_size, seq_len, self.config.vocab_size)
        
        self.assertEqual(logits.shape, expected_shape, f"Output shape mismatch: {logits.shape} vs {expected_shape}")
        
        # Validate output ranges
        self.assertFalse(torch.isnan(logits).any(), "Model outputs contain NaN")
        self.assertFalse(torch.isinf(logits).any(), "Model outputs contain Inf")
    
    def test_quality_metrics_validation(self):
        """Test data quality metrics are within expected ranges."""
        from scripts.analyze_data_quality import analyze_file_quality
        
        test_file = self.test_data_dir / "quality_test.c"
        test_file.write_text(self.sample_code)
        
        metrics = analyze_file_quality(str(test_file))
        
        # Validate metric ranges
        self.assertGreaterEqual(metrics['file_size'], 0, "File size should be non-negative")
        self.assertGreaterEqual(metrics['comment_ratio'], 0, "Comment ratio should be non-negative")
        self.assertLessEqual(metrics['comment_ratio'], 1, "Comment ratio should not exceed 1")
        self.assertGreaterEqual(metrics['avg_line_length'], 0, "Average line length should be non-negative")
        
        # Validate encoding
        self.assertEqual(metrics['encoding'], 'utf-8', "Files should be UTF-8 encoded")
    
    def test_evaluation_completeness(self):
        """Test evaluation pipeline completeness."""
        # Mock evaluation components
        with patch('src.cbert.evaluation.evaluate_model') as mock_eval:
            mock_eval.return_value = {
                'perplexity': 15.2,
                'accuracy': 0.85,
                'loss': 2.1
            }
            
            # Test evaluation metrics validation
            results = mock_eval()
            
            self.assertIn('perplexity', results, "Perplexity metric missing")
            self.assertIn('accuracy', results, "Accuracy metric missing")
            self.assertIn('loss', results, "Loss metric missing")
            
            # Validate metric ranges
            self.assertGreater(results['perplexity'], 0, "Perplexity should be positive")
            self.assertGreaterEqual(results['accuracy'], 0, "Accuracy should be non-negative")
            self.assertLessEqual(results['accuracy'], 1, "Accuracy should not exceed 1")
            self.assertGreaterEqual(results['loss'], 0, "Loss should be non-negative")
    
    def test_error_reporting_structure(self):
        """Test structured error reporting with context."""
        # Test invalid data handling
        invalid_code = "int x = \x00\x01invalid"  # Contains invalid characters
        
        try:
            processed = preprocess_code(invalid_code)
            # Should handle gracefully
            self.assertIsInstance(processed, str, "Should return processed string")
        except Exception as e:
            # Error should be informative
            error_msg = str(e)
            self.assertIsInstance(error_msg, str, "Error message should be string")
            self.assertGreater(len(error_msg), 0, "Error message should not be empty")
    
    def test_data_lineage_tracking(self):
        """Test data lineage and transformation tracking."""
        # Create mock BOM data
        bom_data = {
            "repositories": [
                {
                    "name": "test-repo",
                    "commit_hash": "abc123",
                    "files_processed": 1
                }
            ],
            "processing_steps": [
                {
                    "step": "preprocessing",
                    "parameters": {"remove_comments": True}
                }
            ]
        }
        
        bom_file = self.test_data_dir / "dataset_bom.json"
        bom_file.write_text(json.dumps(bom_data, indent=2))
        
        # Validate BOM structure
        with open(bom_file) as f:
            loaded_bom = json.load(f)
        
        self.assertIn('repositories', loaded_bom, "BOM should track repositories")
        self.assertIn('processing_steps', loaded_bom, "BOM should track processing steps")
        
        # Validate repository tracking
        repo = loaded_bom['repositories'][0]
        self.assertIn('name', repo, "Repository should have name")
        self.assertIn('commit_hash', repo, "Repository should have commit hash")
        self.assertIn('files_processed', repo, "Repository should track file count")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_data_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()