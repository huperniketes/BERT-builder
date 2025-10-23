import argparse
import os
import re

def remove_comments(text):
    """Removes C-style comments from a string."""
    # Remove /* ... */ comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # Remove // ... comments
    text = re.sub(r'//.*', '', text)
    return text

def process_file(file_path):
    """Reads a file, removes comments, and returns the cleaned text."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            # Explicitly remove null bytes, as they can cause issues with text processing
            content = content.replace('\x00', '')
            cleaned_text = remove_comments(content)

            # Normalize UTF-8 encoding while preserving foreign language text in strings
            cleaned_text = cleaned_text.encode('utf-8', errors='replace').decode('utf-8')

            return cleaned_text
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser(description="Pre-process a corpus of C source code by removing comments.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the raw C source code files.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file to save the concatenated, cleaned corpus.")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Starting pre-processing of files in {args.input_dir}...")

    repos = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    total_repos = len(repos)
    
    with open(args.output_file, 'w', encoding='utf-8') as outfile:
        for repo_idx, repo_name in enumerate(sorted(repos)):
            repo_path = os.path.join(args.input_dir, repo_name)
            files_in_repo = [os.path.join(root, file) for root, _, files in os.walk(repo_path) for file in files if file.endswith(('.c', '.h'))]
            total_files_in_repo = len(files_in_repo)

            for file_idx, file_path in enumerate(files_in_repo):
                file_dir = os.path.basename(os.path.dirname(file_path))
                progress_msg = f"Processing Repo {repo_idx + 1}/{total_repos}: {repo_name} | File {file_idx + 1}/{total_files_in_repo}: {file_dir}\x1b[K"
                print(progress_msg, end='\r')

                cleaned_text = process_file(file_path)
                if cleaned_text.strip(): # Only write if there's actual content
                    outfile.write(cleaned_text)
                    outfile.write('\n') # Add a newline between files

    print(f"\nPre-processing complete. Cleaned corpus saved to {args.output_file}")

if __name__ == "__main__":
    main()
