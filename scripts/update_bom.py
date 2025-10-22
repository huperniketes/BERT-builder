import argparse
import json
import os
import requests
import time
import subprocess

def get_latest_commit_hash_remote(repo_full_name, github_token=None):
    """
    Fetches the latest commit hash for a given repository from GitHub API.
    """
    headers = {}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    api_url = f"https://api.github.com/repos/{repo_full_name}/commits?per_page=1"
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        commits = response.json()
        if commits:
            return commits[0]["sha"]
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching commit for {repo_full_name} from API: {e}")
        return None

def get_local_repo_info(repo_path):
    """
    Extracts the latest commit hash and remote URL from a local Git repository.
    """
    commit_hash = None
    remote_url = None
    try:
        # Get commit hash
        result_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        commit_hash = result_hash.stdout.strip()

        # Get remote URL
        result_url = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        remote_url = result_url.stdout.strip()

    except subprocess.CalledProcessError as e:
        print(f"Error getting Git info from {repo_path}: {e.stderr.strip()}")
    except FileNotFoundError:
        print(f"Git command not found. Is Git installed and in PATH?")
    return commit_hash, remote_url

def find_git_repos(search_path, max_depth=1):
    """
    Finds Git repositories within a given search path up to a certain depth.
    Returns a dictionary mapping repo_name to its absolute path.
    """
    found_repos = {}
    for root, dirs, files in os.walk(search_path):
        current_depth = root.count(os.sep) - search_path.count(os.sep)
        if current_depth > max_depth:
            del dirs[:] # Don't recurse further
            continue

        if ".git" in dirs:
            repo_path = root
            repo_name = os.path.basename(repo_path)
            found_repos[repo_name] = repo_path
            dirs.remove(".git") # Don't recurse into .git
        
        # If we are at max_depth, don't go into subdirectories
        if current_depth == max_depth:
            del dirs[:]

    return found_repos

def main():
    parser = argparse.ArgumentParser(description="Extracts detailed information from GitHub API repo data or local Git repos.")
    parser.add_argument("--input_bom", type=str, help="Path to the input dataset BOM file (e.g., top_100_C_repos.json). Required if --update_bom is not used.")
    parser.add_argument("--output_bom", type=str, required=True, help="Path to the output dataset BOM file.")
    parser.add_argument("--github_token", type=str, default=os.environ.get("GITHUB_TOKEN"),
                        help="GitHub personal access token for higher API rate limits (optional).")
    parser.add_argument("--local_path", type=str, default=None,
                        help="Path to a local directory to search for Git repositories. Defaults to current directory if --update_bom is used.")
    parser.add_argument("--update_bom", action="store_true",
                        help="If set, updates the output_bom with local Git info instead of making API calls.")
    args = parser.parse_args()

    if args.update_bom:
        if not os.path.exists(args.output_bom):
            print(f"Error: --update_bom requires an existing output file at {args.output_bom}")
            return
        with open(args.output_bom, 'r', encoding='utf-8') as f:
            extracted_info = json.load(f)

        search_path = args.local_path if args.local_path else os.getcwd()
        print(f"Searching for local Git repositories in {search_path}...")
        local_repos = find_git_repos(search_path, max_depth=1) # Only immediate subdirectories
        print(f"Found {len(local_repos)} local Git repositories.")

        updated_count = 0
        for entry in extracted_info:
            repo_name = entry.get("name")
            repo_url = entry.get("url")
            
            local_repo_path = None
            if repo_name and repo_name in local_repos:
                local_repo_path = local_repos[repo_name]
            elif repo_url:
                # Try to match by URL if name doesn't work or is ambiguous
                for name, path in local_repos.items():
                    _, local_remote_url = get_local_repo_info(path)
                    if local_remote_url and local_remote_url == repo_url:
                        local_repo_path = path
                        break

            if local_repo_path:
                print(f"Updating info for {repo_name} from local repo at {local_repo_path}...")
                commit_hash, remote_url = get_local_repo_info(local_repo_path)
                if commit_hash:
                    entry["commit_hash"] = commit_hash
                    updated_count += 1
                if remote_url and not entry.get("url"): # Only update URL if it's missing
                    entry["url"] = remote_url

        print(f"Updated {updated_count} entries with local Git information.")

    else: # Original behavior: fetch from GitHub API
        if not args.input_bom:
            parser.error("--input_bom is required if --update_bom is not used.")

        if not os.path.exists(args.input_bom):
            print(f"Error: Input file not found at {args.input_bom}")
            return

        with open(args.input_bom, 'r', encoding='utf-8') as f:
            repos_data = json.load(f)

        extracted_info = []
        for i, repo in enumerate(repos_data):
            repo_name = repo.get("name")
            author = repo.get("author")
            url = repo.get("url")
            stars = repo.get("stars")
            language = repo.get("language")
            description = repo.get("description")

            commit_hash = None
            if repo_full_name:
                print(f"[{i+1}/{len(repos_data)}] Fetching commit for {repo_full_name} from API...")
                commit_hash = get_latest_commit_hash_remote(repo_full_name, args.github_token)
                time.sleep(0.1) # Small delay to avoid hitting rate limits too quickly

            extracted_info.append({
                "name": repo_name,
                "author": author,
                "url": url,
                "stars": stars,
                "language": language,
                "description": description,
                "commit_hash": commit_hash
            })

    os.makedirs(os.path.dirname(args.output_bom) or '.', exist_ok=True)
    with open(args.output_bom, 'w', encoding='utf-8') as f:
        json.dump(extracted_info, f, indent=2)

    print(f"Final information for {len(extracted_info)} repositories saved to {args.output_bom}")

if __name__ == "__main__":
    main()
