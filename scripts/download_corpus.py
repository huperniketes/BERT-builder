import argparse
import json
import os
import requests
import subprocess
import time

# Default BOM path relative to the project root
DEFAULT_BOM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "dataset_bom.json")

def get_top_c_repos_from_api(language="C", num_repos=100):
    """Fetches the top starred repositories for a given language from the GitHub API."""
    api_url = f"https://api.github.com/search/repositories?q=language:{language}&sort=stars&order=desc&per_page={num_repos}"
    try:
        print(f"Fetching top {num_repos} starred {language} repositories from GitHub API...")
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        repos = response.json()["items"]
        return repos
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories from API: {e}")
        return []

def get_local_commit_hash(repo_path):
    """
    Gets the HEAD commit hash of a local Git repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting commit hash from {repo_path}: {e.stderr.strip()}")
        return None

def clone_repo(repo_name, clone_url, output_dir, commit_hash=None, depth=None):
    """
    Clones a single repository into the output directory, optionally checking out a specific commit
    or performing a shallow clone.
    """
    repo_path = os.path.join(output_dir, repo_name)

    if os.path.exists(repo_path):
        print(f"Repository {repo_name} already exists. Skipping clone.")
        # If it exists, ensure it's at the correct commit if specified
        if commit_hash:
            current_local_commit = get_local_commit_hash(repo_path)
            if current_local_commit != commit_hash:
                print(f"Checking out commit {commit_hash} for {repo_name}...")
                try:
                    subprocess.run(["git", "checkout", commit_hash], cwd=repo_path, check=True, capture_output=True, text=True)
                    print(f"Successfully checked out {commit_hash} for {repo_name}.")
                except subprocess.CalledProcessError as e:
                    print(f"Error checking out commit {commit_hash} for {repo_name}: {e.stderr}")
            else:
                print(f"Repository {repo_name} is already at commit {commit_hash}.")
        return

    print(f"Cloning {repo_name} from {clone_url}...")
    try:
        git_command = ["git", "clone"]
        if depth:
            git_command.extend(["--depth", str(depth)])
        git_command.extend([clone_url, repo_path])
        
        subprocess.run(git_command, check=True, capture_output=True, text=True)
        print(f"Successfully cloned {repo_name}.")

        if commit_hash and not depth: # If shallow clone, commit_hash might not be available for checkout
            print(f"Checking out commit {commit_hash} for {repo_name}...")
            subprocess.run(["git", "checkout", commit_hash], cwd=repo_path, check=True, capture_output=True, text=True)
            print(f"Successfully checked out {commit_hash} for {repo_name}.")

    except subprocess.CalledProcessError as e:
        print(f"Error cloning or checking out {repo_name}: {e.stderr}")
        if os.path.exists(repo_path):
            try:
                subprocess.run(["rm", "-rf", repo_path], check=True)
                print(f"Cleaned up failed clone attempt for {repo_name}")
            except subprocess.CalledProcessError as cleanup_error:
                print(f"Warning: Failed to clean up {repo_name} directory: {cleanup_error.stderr}")

def main():
    parser = argparse.ArgumentParser(description="Download C repositories from GitHub, either from an existing dataset BOM or initializing one with GitHub's ranking.")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"),
                       help="The directory to clone the repositories into (default: data/ in project root)")
    parser.add_argument("--bom_json", type=str, default=DEFAULT_BOM_PATH,
                        help=f"Path to a dataset BOM (Bill of Materials) with repo details including commit hashes. Defaults to {DEFAULT_BOM_PATH}.")
    parser.add_argument("--init_bom", action="store_true",
                        help="If set, queries GitHub API for top repos, downloads their latest commits, and generates a new dataset BOM.")
    parser.add_argument("--language", type=str, default="C",
                        help="Programming language to query GitHub API for when initializing BOM (default: C).")
    parser.add_argument("--num_repos", type=int, default=100,
                        help="Number of repositories to fetch from GitHub API when initializing BOM (default: 100).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    repos_to_download = []
    bom_to_write = []

    if args.init_bom:
        print("Initializing dataset BOM from GitHub API...")
        api_repos = get_top_c_repos_from_api(args.language, args.num_repos)
        if not api_repos:
            print("Could not fetch repository list from GitHub API. Exiting.")
            return
        
        print(f"\nShallow cloning {len(api_repos)} repositories for their latest commit...")
        for i, repo in enumerate(api_repos):
            repo_name = repo["name"]
            clone_url = repo["clone_url"]
            repo_path = os.path.join(args.output_dir, repo_name)

            print(f"[{i+1}/{len(api_repos)}] Processing {repo_name}...")
            # Perform shallow clone to get the latest commit hash locally
            clone_repo(repo_name, clone_url, args.output_dir, depth=1) # Shallow clone
            
            commit_hash = get_local_commit_hash(repo_path)
            if commit_hash:
                bom_to_write.append({
                    "name": repo_name,
                    "author": repo["owner"]["login"],
                    "url": repo["html_url"],
                    "stars": repo["stargazers_count"],
                    "language": repo["language"],
                    "description": repo["description"],
                    "commit_hash": commit_hash
                })
            else:
                print(f"Warning: Could not get commit hash for {repo_name}. Skipping entry in BOM.")
            time.sleep(0.1) # Small delay to avoid hitting API limits if any subsequent API calls were made

        if bom_to_write:
            with open(args.bom_json, 'w', encoding='utf-8') as f:
                json.dump(bom_to_write, f, indent=2)
            print(f"\nNew BOM written to {args.bom_json} with {len(bom_to_write)} entries.")
            repos_to_download = bom_to_write # Use the newly created BOM for actual download
        else:
            print("No repositories processed for BOM. Exiting.")
            return

    else: # Not initializing BOM, so expect an existing one
        if not os.path.exists(args.bom_json):
            print(f"Error: BOM file not found at {args.bom_json}. Use --init_bom to create one or provide a valid --bom_json path.")
            return
        
        print(f"Loading repository list from BOM: {args.bom_json}...")
        try:
            with open(args.bom_json, 'r', encoding='utf-8') as f:
                bom_data = json.load(f)
            
            for entry in bom_data:
                if "name" in entry and "url" in entry and "commit_hash" in entry:
                    repos_to_download.append({
                        "name": entry["name"],
                        "clone_url": entry["url"],
                        "commit_hash": entry["commit_hash"]
                    })
                else:
                    print(f"Warning: BOM entry missing required fields (name, url, commit_hash): {entry}. Skipping.")

        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading BOM file {args.bom_json}: {e}. Exiting.")
            return

    if not repos_to_download:
        print("No repositories to download. Exiting.")
        return

    print(f"\nStarting download of {len(repos_to_download)} repositories...")
    for repo_info in repos_to_download:
        clone_repo(repo_info["name"], repo_info["clone_url"], args.output_dir, repo_info.get("commit_hash"))

    print("\nCorpus download complete.")

if __name__ == "__main__":
    main()
