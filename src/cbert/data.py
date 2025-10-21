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

            # Character Range Check: Log warning for non-ASCII characters
            non_ascii_chars = [char for char in cleaned_text if ord(char) > 127]
            if non_ascii_chars:
                print(f"Warning: Non-ASCII characters found in {file_path}. Examples: {set(non_ascii_chars)}")

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
    
    with open(args.output_file, 'w', encoding='utf-8') as outfile:
        for root, _, files in os.walk(args.input_dir):
            for file in files:
                if file.endswith(('.c', '.h')):
                    file_path = os.path.join(root, file)
                    print(f"Processing {file_path}...")
                    cleaned_text = process_file(file_path)
                    if cleaned_text.strip(): # Only write if there's actual content
                        outfile.write(cleaned_text)
                        outfile.write('\n') # Add a newline between files

    print(f"\nPre-processing complete. Cleaned corpus saved to {args.output_file}")

if __name__ == "__main__":
    main()
