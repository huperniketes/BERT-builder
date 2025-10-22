import unittest
import os
import json
import shutil
import tempfile
import subprocess
from unittest.mock import patch, MagicMock

class TestDownloadCorpusIntegration(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.test_dir, "repos")
        os.makedirs(self.output_dir)

        # Create a dummy BOM file
        self.bom_path = os.path.join(self.test_dir, "dummy_bom.json")
        self.dummy_repos = [
            {
                "name": "test_repo_1",
                "url": "https://github.com/octocat/Spoon-Knife.git",
                "latest_commit_hash": "9f12700907090909090909090909090909090909" # A real commit from Spoon-Knife
            },
            {
                "name": "test_repo_2",
                "url": "https://github.com/octocat/Hello-World.git",
                "latest_commit_hash": "7630413a95862a29113c765583a005490855ad36" # A real commit from Hello-World
            }
        ]
        with open(self.bom_path, 'w') as f:
            json.dump(self.dummy_repos, f, indent=2)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _run_download_script(self, bom_path=None, output_dir=None):
        command = [
            "python3", "scripts/download_corpus.py",
            "--output_dir", output_dir if output_dir else self.output_dir
        ]
        if bom_path:
            command.extend(["--bom_json", bom_path])
        
        result = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy())
        return result

    def test_download_with_bom(self):
        result = self._run_download_script(bom_path=self.bom_path)
        self.assertEqual(result.returncode, 0, f"Script failed with error: {result.stderr}\n{result.stdout}")

        # Check if repos are cloned and correct commits are checked out
        for repo_info in self.dummy_repos:
            repo_path = os.path.join(self.output_dir, repo_info["name"])
            self.assertTrue(os.path.exists(repo_path))
            self.assertTrue(os.path.isdir(repo_path))

            # Verify commit hash
            current_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            self.assertEqual(current_commit, repo_info["latest_commit_hash"])

    @patch('scripts.download_corpus.get_top_c_repos_from_api')
    def test_download_fallback_to_api(self, mock_get_top_c_repos_from_api):
        # Mock API response if BOM is not found
        mock_get_top_c_repos_from_api.return_value = [
            {
                "name": "mock_repo_api",
                "clone_url": "https://github.com/octocat/Spoon-Knife.git",
                "commit_hash": None # No specific commit hash from API search results
            }
        ]

        # Run without BOM, should fallback to API
        result = self._run_download_script(bom_path=os.path.join(self.test_dir, "non_existent_bom.json"))
        self.assertEqual(result.returncode, 0, f"Script failed with error: {result.stderr}\n{result.stdout}")

        repo_path = os.path.join(self.output_dir, "mock_repo_api")
        self.assertTrue(os.path.exists(repo_path))
        self.assertTrue(os.path.isdir(repo_path))
        mock_get_top_c_repos_from_api.assert_called_once()

    def test_download_existing_repo_with_bom(self):
        # Clone one repo first
        repo_info = self.dummy_repos[0]
        clone_path = os.path.join(self.output_dir, repo_info["name"])
        subprocess.run(["git", "clone", repo_info["url"], clone_path], check=True, capture_output=True, text=True)
        
        # Run download script again with BOM
        result = self._run_download_script(bom_path=self.bom_path)
        self.assertEqual(result.returncode, 0, f"Script failed with error: {result.stderr}\n{result.stdout}")

        # Should have skipped cloning and checked out the commit
        self.assertIn(f"Repository {repo_info["name"]} already exists. Skipping clone.", result.stdout)
        self.assertIn(f"Checking out commit {repo_info["latest_commit_hash"]} for {repo_info["name"]}...", result.stdout)

        # Verify commit hash
        current_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=clone_path,
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()
        self.assertEqual(current_commit, repo_info["latest_commit_hash"])

    def test_download_invalid_bom_format(self):
        invalid_bom_path = os.path.join(self.test_dir, "invalid_bom.json")
        with open(invalid_bom_path, 'w') as f:
            f.write("[{{\"name\": \"repo\", \"url\": \"url\"}}]") # Missing commit hash

        result = self._run_download_script(bom_path=invalid_bom_path)
        self.assertEqual(result.returncode, 0, f"Script failed with error: {result.stderr}\n{result.stdout}")
        self.assertIn("Warning: BOM entry missing required fields", result.stdout)
        self.assertIn("Falling back to GitHub API", result.stdout)

if __name__ == '__main__':
    unittest.main()
