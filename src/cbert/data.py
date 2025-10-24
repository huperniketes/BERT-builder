import argparse
import os
import re
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Any
from .validation import ValidationPipeline
from .semantic_validator import SemanticPreservationValidator

def remove_comments(text):
    """Removes C-style comments from a string."""
    # Remove /* ... */ comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # Remove // ... comments
    text = re.sub(r'//.*', '', text)
    return text

def remove_comments_safe(text):
    """Safely remove comments while preserving string literals and preprocessor directives."""
    result = []
    i = 0
    in_string = False
    in_char = False
    string_char = None
    
    while i < len(text):
        char = text[i]
        
        # Handle string literals
        if not in_char and char in ['"', "'"] and (i == 0 or text[i-1] != '\\'):
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
                string_char = None
            result.append(char)
            i += 1
            continue
        
        # Skip comment removal inside strings
        if in_string:
            result.append(char)
            i += 1
            continue
        
        # Handle /* */ comments
        if i < len(text) - 1 and text[i:i+2] == '/*':
            # Skip to end of comment
            j = i + 2
            while j < len(text) - 1 and text[j:j+2] != '*/':
                j += 1
            if j < len(text) - 1:
                i = j + 2
            else:
                i = len(text)
            result.append(' ')  # Replace with space to preserve structure
            continue
        
        # Handle // comments
        if i < len(text) - 1 and text[i:i+2] == '//':
            # Skip to end of line
            while i < len(text) and text[i] != '\n':
                i += 1
            continue
        
        result.append(char)
        i += 1
    
    return ''.join(result)

def normalize_whitespace(text):
    """Normalize whitespace while preserving code structure."""
    # Preserve single spaces around operators and keywords
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    text = re.sub(r'\s*([{}();,])\s*', r'\1', text)  # Remove spaces around structural chars
    text = re.sub(r'\s*([=+\-*/%<>!&|^])\s*', r' \1 ', text)  # Normalize operator spacing
    return text.strip()

def preprocess_code(code: str) -> str:
    """Preprocess C code by removing comments and normalizing while preserving semantics."""
    # Remove comments but preserve string literals and preprocessor directives
    cleaned = remove_comments_safe(code)
    # Remove null bytes
    cleaned = cleaned.replace('\x00', '')
    # Normalize whitespace while preserving structure
    cleaned = normalize_whitespace(cleaned)
    # Normalize UTF-8 encoding
    cleaned = cleaned.encode('utf-8', errors='replace').decode('utf-8')
    return cleaned

class CCodeDataset(Dataset):
    """Dataset for C code files."""
    
    def __init__(self, data_dir: str, tokenizer, max_length: int = 512):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.files = list(self.data_dir.glob('**/*.c')) + list(self.data_dir.glob('**/*.h'))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        file_path = self.files[idx]
        code = process_file(str(file_path))
        
        # Tokenize
        tokens = self.tokenizer.encode(code)
        
        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones(len(tokens), dtype=torch.long)
        
        # Pad if necessary
        if len(tokens) < self.max_length:
            pad_length = self.max_length - len(tokens)
            input_ids = torch.cat([input_ids, torch.zeros(pad_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

def process_file(file_path):
    """Reads a file, removes comments, and returns the cleaned text with semantic validation."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            # Explicitly remove null bytes, as they can cause issues with text processing
            content = content.replace('\x00', '')
            
            # Use semantic-aware preprocessing
            cleaned_text = preprocess_code(content)
            
            # Validate semantic preservation for critical files
            if len(content) > 100:  # Only validate non-trivial files
                validator = SemanticPreservationValidator()
                issues = validator.validate_preprocessing(content, cleaned_text)
                
                critical_issues = [i for i in issues if i.severity == 'critical']
                if critical_issues:
                    print(f"WARNING: Critical semantic issues in {file_path}: {len(critical_issues)} issues")

            return cleaned_text
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser(description="Pre-process a corpus of C source code by removing comments.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the raw C source code files.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file to save the concatenated, cleaned corpus.")
    parser.add_argument("--validation_dir", type=str, help="Directory to save validation results.")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize validation pipeline
    validation_dir = args.validation_dir or os.path.join(output_dir, "validation")
    validator = ValidationPipeline(validation_dir)
    semantic_validator = SemanticPreservationValidator()

    print(f"Starting pre-processing of files in {args.input_dir}...")

    repos = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    total_repos = len(repos)
    
    line_offsets = []
    current_offset = 0
    
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
                    line_offsets.append(current_offset)
                    line_content = cleaned_text + '\n'
                    outfile.write(line_content)
                    current_offset += len(line_content.encode('utf-8'))

    print(f"\nPre-processing complete. Cleaned corpus saved to {args.output_file}")
    
    # Save line offsets if requested
    lineoffsets_file = args.output_file.replace('.txt', '_lineoffsets.json')
    import json
    with open(lineoffsets_file, 'w') as f:
        json.dump({'line_offsets': line_offsets}, f)
    print(f"Line offsets saved to {lineoffsets_file}")
    
    # Validate preprocessing results
    validation_result = validator.validate_data_preprocessing(args.input_dir, args.output_file)
    if not validation_result.passed:
        print(f"WARNING: Data preprocessing validation failed with {len(validation_result.issues)} issues")
    else:
        print("Data preprocessing validation passed")

if __name__ == "__main__":
    main()
