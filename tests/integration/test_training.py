import unittest
import os
import json
import shutil
import tempfile
from cbert import trainer
from cbert.tokenizer import CharTokenizer, KeyCharTokenizer, SentencePieceTokenizer

class Args:
    def __init__(self, dataset_dir, config, tokenizer_name, output_dir, masking='mlm', max_steps=10, resume_from_checkpoint=None, batch_size=2, learning_rate=2e-5, vocab_file=None, spm_model_file=None):
        self.dataset_dir = dataset_dir
        self.config = config
        self.tokenizer = tokenizer_name
        self.output_dir = output_dir
        self.epochs = 1
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.resume_from_checkpoint = resume_from_checkpoint
        self.masking = masking
        self.vocab_file = vocab_file
        self.spm_model_file = spm_model_file

class TestTrainerIntegration(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # Create dummy dataset
        self.dataset_path = os.path.join(self.test_dir, "dataset.txt")
        with open(self.dataset_path, "w") as f:
            for i in range(20):
                f.write(f"int main_{i}() {{ return {i}; }}\n")
            f.write("// This is a comment only line\n") # Should be filtered
            f.write("\n") # Empty line, should be filtered
            f.write("char *s = \"résumé\";\n") # Non-ASCII, should warn

        # Create dummy config
        self.config_path = os.path.join(self.test_dir, "config.json")
        config = {
            "vocab_size": 128, # CharTokenizer vocab size (will be resized by model)
            "hidden_size": 64, # smaller for faster testing
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 128,
            "max_position_embeddings": 128
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f)

        # Create dummy SentencePiece model files
        self.spm_model_prefix = os.path.join(self.test_dir, "test_spm")
        SentencePieceTokenizer.train(self.dataset_path, self.spm_model_prefix, vocab_size=50)
        self.spm_model_file = self.spm_model_prefix + ".model"
        self.spm_vocab_file = self.spm_model_prefix + ".json"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _run_training_test(self, tokenizer_name, masking_strategy, max_steps, expected_log_entries, vocab_file=None, spm_model_file=None):
        output_dir = os.path.join(self.test_dir, f"output_{tokenizer_name}_{masking_strategy}")
        args = Args(
            dataset_dir=self.dataset_path,
            config=self.config_path,
            tokenizer_name=tokenizer_name,
            output_dir=output_dir,
            masking=masking_strategy,
            max_steps=max_steps,
            vocab_file=vocab_file,
            spm_model_file=spm_model_file
        )
        trainer.run(args)

        # Check for log file
        log_file = os.path.join(output_dir, 'training_log.json')
        self.assertTrue(os.path.exists(log_file))
        with open(log_file, 'r') as f:
            log_entries = [json.loads(line) for line in f]
        self.assertEqual(len(log_entries), expected_log_entries)
        self.assertEqual(log_entries[-1]['step'], expected_log_entries - 1)

        # Check for model and tokenizer saving
        self.assertTrue(os.path.exists(os.path.join(output_dir, 'model.safetensors')) or os.path.exists(os.path.join(output_dir, 'pytorch_model.bin')))
        self.assertTrue(os.path.exists(os.path.join(output_dir, 'vocab.json')) or os.path.exists(os.path.join(output_dir, 'tokenizer.json')))

        # Check for reasonable loss and accuracy (should decrease/increase over steps)
        # This is a heuristic check, not a strict value assertion
        self.assertLess(log_entries[-1]['loss'], log_entries[0]['loss'] * 1.5) # Loss should not explode
        self.assertGreaterEqual(log_entries[-1]['accuracy'], 0.0) # Accuracy should be non-negative

        return output_dir

    def test_training_char_mlm(self):
        self._run_training_test(tokenizer_name='char', masking_strategy='mlm', max_steps=5, expected_log_entries=5)

    def test_training_char_wwm(self):
        self._run_training_test(tokenizer_name='char', masking_strategy='wwm', max_steps=5, expected_log_entries=5)

    def test_training_keychar_mlm(self):
        self._run_training_test(tokenizer_name='keychar', masking_strategy='mlm', max_steps=5, expected_log_entries=5)

    def test_training_keychar_wwm(self):
        self._run_training_test(tokenizer_name='keychar', masking_strategy='wwm', max_steps=5, expected_log_entries=5)

    def test_training_spe_mlm(self):
        self._run_training_test(tokenizer_name='spe', masking_strategy='mlm', max_steps=5, expected_log_entries=5,
                                vocab_file=self.spm_vocab_file, spm_model_file=self.spm_model_file)

    def test_training_spe_wwm(self):
        self._run_training_test(tokenizer_name='spe', masking_strategy='wwm', max_steps=5, expected_log_entries=5,
                                vocab_file=self.spm_vocab_file, spm_model_file=self.spm_model_file)

    def test_training_resumption(self):
        # First run
        output_dir_first = os.path.join(self.test_dir, "output_resume_first")
        args1 = Args(
            dataset_dir=self.dataset_path,
            config=self.config_path,
            tokenizer_name='char',
            output_dir=output_dir_first,
            masking='mlm',
            max_steps=3 # Run for 3 steps
        )
        trainer.run(args1)

        # Check for checkpoint
        checkpoint_dir = os.path.join(output_dir_first, 'checkpoint-2') # Checkpoint at step 2 (0-indexed)
        self.assertTrue(os.path.exists(checkpoint_dir))
        self.assertTrue(os.path.exists(os.path.join(checkpoint_dir, 'training_state.bin')))

        # Second run (resuming)
        output_dir_second = output_dir_first # Use the same output dir for final model
        args2 = Args(
            dataset_dir=self.dataset_path,
            config=self.config_path,
            tokenizer_name='char',
            output_dir=output_dir_second,
            masking='mlm',
            max_steps=5, # Total steps for second run
            resume_from_checkpoint=checkpoint_dir
        )
        trainer.run(args2)

        # Check if log file was appended and continued from step 3
        log_file = os.path.join(output_dir_second, 'training_log.json')
        self.assertTrue(os.path.exists(log_file))
        with open(log_file, 'r') as f:
            log_entries = [json.loads(line) for line in f]
        self.assertEqual(len(log_entries), 5) # Should have 5 entries (steps 0-4)
        self.assertEqual(log_entries[0]['step'], 0) # First logged step should be 0 (from first run)
        self.assertEqual(log_entries[3]['step'], 3) # First resumed step should be 3 (resumed from 2+1=3)
        self.assertEqual(log_entries[-1]['step'], 4) # Last logged step should be 4

        # Check final model saving
        self.assertTrue(os.path.exists(os.path.join(output_dir_second, 'model.safetensors')) or os.path.exists(os.path.join(output_dir_second, 'pytorch_model.bin')))
        self.assertTrue(os.path.exists(os.path.join(output_dir_second, 'vocab.json')) or os.path.exists(os.path.join(output_dir_second, 'tokenizer.json')))

if __name__ == '__main__':
    unittest.main()