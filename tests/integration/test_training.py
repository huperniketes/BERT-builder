import unittest
import os
import json
import shutil
from cbert import trainer

class Args:
    def __init__(self, dataset_path, config_path, tokenizer, output_dir, max_steps=10, resume_from_checkpoint=None):
        self.dataset_path = dataset_path
        self.config_path = config_path
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.epochs = 1
        self.batch_size = 2
        self.learning_rate = 5e-5
        self.max_steps = max_steps
        self.resume_from_checkpoint = resume_from_checkpoint

class TestTrainer(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_trainer_output"
        os.makedirs(self.test_dir, exist_ok=True)

        # Create dummy dataset
        self.dataset_path = os.path.join(self.test_dir, "dataset.txt")
        with open(self.dataset_path, "w") as f:
            for i in range(20):
                f.write(f"int main_{i}() {{ return {i}; }}\n")

        # Create dummy config
        self.config_path = os.path.join(self.test_dir, "config.json")
        config = {
            "vocab_size": 128, # CharTokenizer vocab size
            "hidden_size": 256, # smaller for testing
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 512,
            "max_position_embeddings": 128
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_training_and_resumption(self):
        # First run
        args1 = Args(
            dataset_path=self.dataset_path,
            config_path=self.config_path,
            tokenizer='char',
            output_dir=self.test_dir,
            max_steps=6
        )
        trainer.run(args1)

        # Check for log file
        log_file = os.path.join(self.test_dir, 'training_log.json')
        self.assertTrue(os.path.exists(log_file))
        with open(log_file, 'r') as f:
            log_entries = [json.loads(line) for line in f]
        self.assertEqual(len(log_entries), 6)
        self.assertEqual(log_entries[-1]['step'], 5)


        # Check for checkpoint
        checkpoint_dir = os.path.join(self.test_dir, 'checkpoint-5')
        self.assertTrue(os.path.exists(checkpoint_dir))
        self.assertTrue(os.path.exists(os.path.join(checkpoint_dir, 'model.safetensors')))
        self.assertTrue(os.path.exists(os.path.join(checkpoint_dir, 'training_state.bin')))

        # Second run (resuming)
        args2 = Args(
            dataset_path=self.dataset_path,
            config_path=self.config_path,
            tokenizer='char',
            output_dir=self.test_dir,
            max_steps=10,
            resume_from_checkpoint=checkpoint_dir
        )
        trainer.run(args2)

        # Check if log file was appended
        with open(log_file, 'r') as f:
            log_entries = [json.loads(line) for line in f]
        self.assertEqual(len(log_entries), 10)
        self.assertEqual(log_entries[-1]['step'], 9)


if __name__ == '__main__':
    unittest.main()