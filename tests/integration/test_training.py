import unittest
import os
import subprocess
import json

class TestTraining(unittest.TestCase):

    def setUp(self):
        self.temp_dir = "tests/temp_training"
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Create a dummy dataset
        self.corpus_path = os.path.join(self.data_dir, "dummy_corpus.txt")
        with open(self.corpus_path, "w") as f:
            f.write("int main() { return 0; }\n" * 100)

        # Create a dummy model config
        self.config_path = os.path.join(self.temp_dir, "config.json")
        config = {
            "vocab_size": 128,
            "hidden_size": 128, # Smaller for test
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 512
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            subprocess.run(["rm", "-rf", self.temp_dir])

    def test_training_script(self):
        """Test that the training script runs for one step and creates a checkpoint."""
        command = [
            "python3", "src/cli/train.py",
            "--dataset_path", self.corpus_path,
            "--config_path", self.config_path,
            "--tokenizer", "char",
            "--output_dir", self.output_dir,
            "--epochs", "1",
            "--batch_size", "2",
            "--max_steps", "1" # Run only for a single step
        ]

        # This will fail as the script doesn't exist yet
        result = subprocess.run(command, capture_output=True, text=True)

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        self.assertEqual(result.returncode, 0, f"Training script failed with error: {result.stderr}")
        
        # Check if a checkpoint was created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "model.safetensors")))

if __name__ == '__main__':
    unittest.main()
