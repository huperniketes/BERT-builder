import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizer

from .tokenizer import CharTokenizer, KeyCharTokenizer, SentencePieceTokenizer

class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        # Tokenize the line
        tokens = self.tokenizer.tokenize(line)
        # Convert tokens to IDs
        # This part is still a bit of a placeholder as we don't have a real vocab mapping in the tokenizers
        # For now, we'll use a simple char-to-int mapping for the test
        if isinstance(self.tokenizer, CharTokenizer):
            token_ids = [ord(c) for c in tokens]
        else: # Fallback for other tokenizers in this test
            token_ids = [ord(c) for c in line]


        # Pad or truncate
        if len(token_ids) < self.max_length:
            token_ids += [0] * (self.max_length - len(token_ids)) # Assuming pad token id is 0
        else:
            token_ids = token_ids[:self.max_length]

        input_ids = torch.tensor(token_ids)
        return {"input_ids": input_ids, "labels": input_ids.clone()}

def get_tokenizer(tokenizer_name: str, config: BertConfig) -> PreTrainedTokenizer:
    """
    This function is still a placeholder.
    A real implementation would load a trained tokenizer.
    """
    if tokenizer_name == 'char':
        tok = CharTokenizer()
        tok.vocab_size = config.vocab_size
        return tok
    elif tokenizer_name == 'keychar':
        tok = KeyCharTokenizer()
        tok.vocab_size = config.vocab_size
        return tok
    # SentencePiece requires a model file, which we don't have in the test
    # so we can't instantiate it here yet.
    else:
        raise ValueError(f"Tokenizer '{tokenizer_name}' not supported in this test.")


def run(args):
    """Main training function."""
    print("Starting training run...")

    # 1. Load Config
    with open(args.config_path, 'r') as f:
        config_dict = json.load(f)
    config = BertConfig(**config_dict)

    # 2. Get Tokenizer
    tokenizer = get_tokenizer(args.tokenizer, config)

    # 3. Create Model
    model = BertForMaskedLM(config=config)

    # 4. Create Dataset and DataLoader
    dataset = TextDataset(args.dataset_path, tokenizer, config.max_position_embeddings)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # 5. Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # 6. Training loop
    model.train()
    for i, batch in enumerate(dataloader):
        if args.max_steps > 0 and i >= args.max_steps:
            break
        
        optimizer.zero_grad()
        
        outputs = model(**batch)
        loss = outputs.loss
        
        print(f"Step {i}, Loss: {loss.item()}")

        loss.backward()
        optimizer.step()

    # 7. Save model
    print(f"Saving model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    print(f"Contents of output dir: {os.listdir(args.output_dir)}")
    print("Training run finished.")
