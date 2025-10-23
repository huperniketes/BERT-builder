import argparse
import json
import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForMaskedLM, BertConfig

from cbert.trainer import TextDataset, get_tokenizer # Re-use TextDataset and get_tokenizer
from cbert.model import create_cbert_model # Import create_cbert_model
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_accuracy_mlm(logits, labels):
    predictions = torch.argmax(logits, dim=-1)
    # Only calculate accuracy on non-ignored tokens (labels != -100)
    masked_tokens = labels != -100
    correct = (predictions[masked_tokens] == labels[masked_tokens]).sum().item()
    total = masked_tokens.sum().item()
    return correct / total if total > 0 else 0

def calculate_perplexity(loss):
    return torch.exp(loss).item()

# --- Evaluation Dataset Classes ---
class BaseEvaluationDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.line_offsets = []
        self._build_line_offsets()

    def _build_line_offsets(self):
        current_offset = 0
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                stripped_line = line.strip()
                if not stripped_line: # Skip empty lines
                    current_offset += len(line.encode('utf-8')) # Account for skipped line's bytes
                    continue
                
                # Basic check for meaningful content (non-special tokens)
                encoded_content = self.tokenizer.encode(stripped_line, add_special_tokens=False, max_length=self.max_length, truncation=True)
                if encoded_content: # If there are any actual content tokens
                    self.line_offsets.append(current_offset)
                current_offset += len(line.encode('utf-8'))
        logger.info(f"Loaded {len(self.line_offsets)} meaningful lines from {self.file_path}")

    def __len__(self):
        return len(self.line_offsets)

    def _get_line(self, idx):
        offset = self.line_offsets[idx]
        with open(self.file_path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            return f.readline().strip()

class MLMEvaluationDataset(BaseEvaluationDataset):
    def __getitem__(self, idx):
        line = self._get_line(idx)
        
        # For MLM evaluation, we need input_ids and labels (original token_ids)
        # We don't apply masking here, as the model will predict based on the input
        # and we compare against the original tokens as labels.
        token_ids = self.tokenizer.encode(line, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True)
        
        input_ids = torch.tensor(token_ids)
        labels = input_ids.clone() # Labels are the original tokens for MLM evaluation

        return {"input_ids": input_ids, "labels": labels}

class ASTEvaluationDataset(BaseEvaluationDataset):
    def __getitem__(self, idx):
        # Placeholder for AST tagging data loading
        # In a real scenario, each line might be a JSON object or a specific format
        # containing code tokens and corresponding AST labels.
        line = self._get_line(idx)
        logger.warning(f"ASTEvaluationDataset is a placeholder. Returning dummy data for line: {line[:50]}...")
        
        input_ids = self.tokenizer.encode(line, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True)
        labels = torch.randint(0, 10, (len(input_ids),)) # Dummy AST labels

        return {"input_ids": torch.tensor(input_ids), "labels": labels}

class VIEvaluationDataset(BaseEvaluationDataset):
    def __getitem__(self, idx):
        # Placeholder for Vulnerability Identification data loading
        # In a real scenario, each line might be a code snippet and a binary label (0/1).
        line = self._get_line(idx)
        logger.warning(f"VIEvaluationDataset is a placeholder. Returning dummy data for line: {line[:50]}...")

        input_ids = self.tokenizer.encode(line, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True)
        label = torch.tensor(random.randint(0, 1)) # Dummy binary label

        return {"input_ids": torch.tensor(input_ids), "labels": label}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a C-BERT model.")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to the trained model directory (Hugging Face format). This directory should contain config.json, model.safetensors/pytorch_model.bin, and tokenizer files.")
    parser.add_argument("--task", type=str, default='mlm', choices=['mlm', 'ast', 'vi'], help="The evaluation task.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to the pre-processed evaluation data file.")
    parser.add_argument("--output-file", type=str, default=None, help="File to save the JSON evaluation results (default: stdout).")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum sequence length for tokenization.")
    parser.add_argument("--tokenizer-type", type=str, required=True, choices=['char', 'keychar', 'spe'], help="Type of tokenizer used during training.")
    parser.add_argument("--vocab-file", type=str, default=None, help="Path to the vocabulary file (required for SentencePiece tokenizer). If not provided, attempts to load from model-dir.")
    parser.add_argument("--spm-model-file", type=str, default=None, help="Path to the SentencePiece model file (required for SentencePiece tokenizer). If not provided, attempts to load from model-dir.")

    args = parser.parse_args()

    # Validate SentencePiece tokenizer requirements
    if args.tokenizer_type == 'spe':
        if not args.vocab_file and not os.path.exists(os.path.join(args.model_dir, "vocab.json")):
            print("Error: --vocab-file is required for SentencePiece tokenizer or vocab.json must exist in model-dir")
            sys.exit(1)
        if not args.spm_model_file and not os.path.exists(os.path.join(args.model_dir, "spm.model")):
            print("Error: --spm-model-file is required for SentencePiece tokenizer or spm.model must exist in model-dir")
            sys.exit(1)

    # Validate file paths exist
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory does not exist: {args.model_dir}")
        sys.exit(1)
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory does not exist: {args.dataset_dir}")
        sys.exit(1)
    if args.vocab_file and not os.path.exists(args.vocab_file):
        print(f"Error: Vocab file does not exist: {args.vocab_file}")
        sys.exit(1)
    if args.spm_model_file and not os.path.exists(args.spm_model_file):
        print(f"Error: SentencePiece model file does not exist: {args.smp_model_file}")
        sys.exit(1)

    # Validate numeric arguments
    if args.batch_size <= 0:
        print("Error: --batch-size must be greater than 0")
        sys.exit(1)
    if args.max_length <= 0:
        print("Error: --max-length must be greater than 0")
        sys.exit(1)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Model Configuration
    config = BertConfig.from_pretrained(args.model_dir)
    if args.max_length != config.max_position_embeddings:
        logger.warning(f"Provided max_length ({args.max_length}) differs from model config max_position_embeddings ({config.max_position_embeddings}). Using config value.")
        args.max_length = config.max_position_embeddings

    # 2. Load Tokenizer
    vocab_file = args.vocab_file if args.vocab_file else os.path.join(args.model_dir, "vocab.json")
    spm_model_file = args.spm_model_file if args.spm_model_file else os.path.join(args.model_dir, "spm.model")
    
    tokenizer = get_tokenizer(args.tokenizer_type, vocab_file=vocab_file, spm_model_file=spm_model_file)
    logger.info(f"Tokenizer loaded: {type(tokenizer).__name__}")

    # 3. Load Model
    model = BertForMaskedLM.from_pretrained(args.model_dir, config=config)
    model.to(device)
    model.eval() # Set model to evaluation mode
    logger.info("Model loaded and set to evaluation mode.")

    # 4. Load Dataset and DataLoader
    if args.task == 'mlm':
        eval_dataset = MLMEvaluationDataset(args.dataset_dir, tokenizer, args.max_length)
    elif args.task == 'ast':
        eval_dataset = ASTEvaluationDataset(args.dataset_dir, tokenizer, args.max_length)
    elif args.task == 'vi':
        eval_dataset = VIEvaluationDataset(args.dataset_dir, tokenizer, args.max_length)
    else:
        raise ValueError(f"Unknown evaluation task: {args.task}")

    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False) # No shuffling for evaluation
    logger.info(f"Evaluation dataset loaded with {len(eval_dataset)} samples.")

    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    # 5. Perform Evaluation
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            
            # Calculate accuracy based on task
            if args.task == 'mlm':
                total_accuracy += calculate_accuracy_mlm(logits, labels)
            elif args.task == 'ast':
                # Placeholder for AST accuracy calculation
                logger.warning("AST accuracy calculation is a placeholder.")
                total_accuracy += 0.0 # Dummy accuracy
            elif args.task == 'vi':
                # Placeholder for VI accuracy calculation
                logger.warning("VI accuracy calculation is a placeholder.")
                total_accuracy += 0.0 # Dummy accuracy

            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0
    avg_perplexity = calculate_perplexity(torch.tensor(avg_loss)) if args.task == 'mlm' else 0.0 # Perplexity only for MLM

    # 6. Calculate Metrics
    results = {
        "task": args.task,
        "average_loss": avg_loss,
        "average_accuracy": avg_accuracy,
        "average_perplexity": avg_perplexity,
        "num_samples": len(eval_dataset)
    }
    logger.info(f"Evaluation Results: {results}")

    # 7. Save Results
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:  # Only create directory if there is one
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to {args.output_file}")
    else:
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
