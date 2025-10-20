import unittest
import subprocess
import json
import os

class TestEmbedCLI(unittest.TestCase):

    def test_embed_cli_output(self):
        dummy_c_code = "int main() { return 0; }"
        command = [
            "python3", "src/cli/embed.py",
            "--code", dummy_c_code
        ]
        
        # Run the command and capture output
        result = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy())

        # Assert that the script ran without errors
        self.assertEqual(result.returncode, 0, f"Script failed with error: {result.stderr}")

        # Assert that stdout contains a JSON array (representing the embedding)
        try:
            embedding = json.loads(result.stdout.strip())
            self.assertIsInstance(embedding, list)
            self.assertGreater(len(embedding), 0)
            self.assertIsInstance(embedding[0], float)
        except json.JSONDecodeError:
            self.fail(f"Output is not valid JSON: {result.stdout}")

if __name__ == '__main__':
    unittest.main()