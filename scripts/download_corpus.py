import argparse
import json
import os
import requests
import subprocess

def get_top_c_repos():
    """Gets the top 100 starred C repositories from the GitHub API."""
    api_url = "https://api.github.com/search/repositories?q=language:C&sort=stars&order=desc&per_page=100"
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()["items"]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}")
        return []

def clone_repo(repo, output_dir):
    """Clones a single repository into the output directory."""
    repo_name = repo["name"]
    clone_url = repo["clone_url"]
    repo_path = os.path.join(output_dir, repo_name)

    if os.path.exists(repo_path):
        print(f"Repository {repo_name} already exists. Skipping.")
        return

    print(f"Cloning {repo_name} from {clone_url}...")
    try:
        subprocess.run(["git", "clone", clone_url, repo_path], check=True, capture_output=True, text=True)
        print(f"Successfully cloned {repo_name}.")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning {repo_name}: {e.stderr}")

def main():
    parser = argparse.ArgumentParser(description="Download the top 100 starred C repositories from GitHub.")
    parser.add_argument("--output_dir", type=str, required=True, help="The directory to clone the repositories into.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Fetching top 100 C repositories...")
    repos = get_top_c_repos()

    if not repos:
        print("Could not fetch repository list. Exiting.")
        return

    for repo in repos:
        clone_repo(repo, args.output_dir)

    print("\nCorpus download complete.")

if __name__ == "__main__":
    main()
