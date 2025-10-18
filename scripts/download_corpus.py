import argparse
import json
import os
import requests
import subprocess

def get_top_c_repos():
    """Gets the top 100 starred C repositories from the GitHub API or loads from cache."""
    cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "top_100_C_repos.json")
    
    # Try to load from cache first
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                print("Loading repository list from cache...")
                return json.loads(f.read())
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading cache file: {e}")
            # Continue to fetch from API if cache read fails
    
    # Fetch from GitHub API if no cache exists
    api_url = "https://api.github.com/search/repositories?q=language:C&sort=stars&order=desc&per_page=100"
    try:
        print("Fetching repository list from GitHub API...")
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        repos = response.json()["items"]
        
        # Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(repos, f, indent=2)
        print("Saved repository list to cache.")
        
        return repos
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
        if os.path.exists(repo_path):
            try:
                subprocess.run(["rm", "-rf", repo_path], check=True)
                print(f"Cleaned up failed clone attempt for {repo_name}")
            except subprocess.CalledProcessError as cleanup_error:
                print(f"Warning: Failed to clean up {repo_name} directory: {cleanup_error.stderr}")

def main():
    parser = argparse.ArgumentParser(description="Download the top 100 starred C repositories from GitHub.")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"),
                       help="The directory to clone the repositories into (default: data/ in project root)")
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
