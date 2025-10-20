
import unittest
import os
import json
import shutil
import subprocess

class TestEvaluationIntegration(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_evaluation_output"
        os.makedirs(self.test_dir, exist_ok=True)

        # Create a dummy model checkpoint directory
        self.model_dir = os.path.join(self.test_dir, "dummy_model")
        os.makedirs(self.model_dir, exist_ok=True)
        # Create a dummy model.safetensors and config.json
        with open(os.path.join(self.model_dir, "model.safetensors"), "w") as f:
            f.write("dummy model weights")
        with open(os.path.join(self.model_dir, "config.json"), "w") as f:
            json.dump({"vocab_size": 128}, f)

        # Create a dummy dataset for evaluation
        self.dataset_path = os.path.join(self.test_dir, "eval_dataset.txt")
        with open(self.dataset_path, "w") as f:
            f.write("int main() { return 0; }\n")
            f.write("int x = 1;\n")

        self.output_file = os.path.join(self.test_dir, "eval_results.json")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_evaluate_script_runs(self):
        # This test will fail until evaluate.py is implemented
        command = [
            "python3", "src/cli/evaluate.py",
            "--model-dir", self.model_dir,
            "--task", "ast", # Dummy task for now
            "--dataset-dir", self.dataset_path,
            "--output-file", self.output_file
        ]
        result = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy())

        # Assert that the script ran without errors (exit code 0)
        self.assertEqual(result.returncode, 0, f"Script failed with error: {result.stderr}")

        # Assert that the output file was created
        self.assertTrue(os.path.exists(self.output_file))

        # Assert that the output file contains valid JSON
        with open(self.output_file, 'r') as f:
            results = json.load(f)
        self.assertIn("accuracy", results)
        self.assertIn("perplexity", results)

if __name__ == '__main__':
    unittest.main()
