import unittest
import subprocess
import json
import os
import tempfile
import shutil
from cbert import trainer
from cbert.tokenizer import CharTokenizer, SentencePieceTokenizer
from transformers import BertConfig

# Helper class for trainer args (re-using from test_training.py)
class TrainerArgs:
    def __init__(self, dataset_dir, config, tokenizer_name, output_dir, masking='mlm', max_steps=5, batch_size=2, learning_rate=2e-5, vocab_file=None, spm_model_file=None):
        self.dataset_dir = dataset_dir
        self.config = config
        self.tokenizer = tokenizer_name
        self.output_dir = output_dir
        self.epochs = 1
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.resume_from_checkpoint = None
        self.masking = masking
        self.vocab_file = vocab_file
        self.spm_model_file = spm_model_file

class TestEmbedCLI(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # Create dummy dataset for training a model
        self.dataset_path = os.path.join(self.test_dir, "dataset.txt")
        with open(self.dataset_path, "w") as f:
            for i in range(20):
                f.write(f"int main_{i}() {{ return {i}; }}\n")

        # Create dummy config
        self.config_path = os.path.join(self.test_dir, "config.json")
        config_dict = {
            "vocab_size": 128, # Will be resized by model
            "hidden_size": 64, # smaller for faster testing
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 128,
            "max_position_embeddings": 128
        }
        with open(self.config_path, "w") as f:
            json.dump(config_dict, f)

        # Train a small model for embedding
        self.trained_model_dir = os.path.join(self.test_dir, "trained_model")
        trainer_args = TrainerArgs(
            dataset_dir=self.dataset_path,
            config=self.config_path,
            tokenizer_name='char',
            output_dir=self.trained_model_dir,
            masking='mlm',
            max_steps=5 # Train for a few steps
        )
        trainer.run(trainer_args)

        # Create dummy SentencePiece model files for testing
        self.spm_model_prefix = os.path.join(self.test_dir, "test_spm")
        SentencePieceTokenizer.train(self.dataset_path, self.spm_model_prefix, vocab_size=50)
        self.spm_model_file = self.spm_model_prefix + ".model"
        self.spm_vocab_file = self.spm_model_prefix + ".vocab"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _run_embed_test(self, tokenizer_type, code_snippet, vocab_file=None, spm_model_file=None):
        output_file = os.path.join(self.test_dir, f"embedding_{tokenizer_type}.json")
        
        command = [
            "python3", "src/cli/embed.py",
            "--code", code_snippet,
            "--model-dir", self.trained_model_dir,
            "--output-file", output_file,
            "--tokenizer-type", tokenizer_type,
            "--max-length", "128",
        ]
        if vocab_file: # Add vocab_file if provided
            command.extend(["--vocab-file", vocab_file])
        if spm_model_file: # Add spm_model_file if provided
            command.extend(["--spm-model-file", spm_model_file])

        result = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy())

        # Assert that the script ran without errors (exit code 0)
        self.assertEqual(result.returncode, 0, f"Script failed with error: {result.stderr}\n{result.stdout}")

        # Assert that the output file was created
        self.assertTrue(os.path.exists(output_file))

        # Assert that the output file contains valid JSON and is a list of floats
        with open(output_file, 'r') as f:
            embedding = json.load(f)
        
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)
        self.assertIsInstance(embedding[0], float)

        # Load the config to get hidden_size for embedding dimension check
        config = BertConfig.from_pretrained(self.trained_model_dir)
        self.assertEqual(len(embedding), config.hidden_size) # Embedding dimension should match hidden_size

    def test_embed_char_tokenizer(self):
        self._run_embed_test(tokenizer_type='char', code_snippet="int main() { return 0; }")

    def test_embed_keychar_tokenizer(self):
        self._run_embed_test(tokenizer_type='keychar', code_snippet="if (x == 0) { return; }")

    def test_embed_spe_tokenizer(self):
        self._run_embed_test(tokenizer_type='spe', code_snippet="void func(int a);",
                                vocab_file=self.spm_vocab_file, spm_model_file=self.spm_model_file)

if __name__ == '__main__':
    unittest.main()