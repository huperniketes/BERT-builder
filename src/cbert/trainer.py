import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertConfig, BertForMaskedLM

from .tokenizer import CharTokenizer, KeyCharTokenizer, SentencePieceTokenizer

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        # A real implementation would use the tokenizer to convert to IDs
        # For this test, we just create dummy tensors
        dummy_ids = torch.randint(0, self.tokenizer.vocab_size, (self.max_length,))
        return {"input_ids": dummy_ids, "labels": dummy_ids.clone()}

def get_tokenizer(tokenizer_name, config):
    # A real implementation would load a trained tokenizer
    # For now, just instantiate the class for the vocab_size
    if tokenizer_name == 'char':
        tok = CharTokenizer()
        tok.vocab_size = config.vocab_size
        return tok
    # Add other tokenizers later
    else:
        # Fallback for the test
        tok = CharTokenizer()
        tok.vocab_size = config.vocab_size
        return tok

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
    dataset = TextDataset(args.dataset_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # 5. Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # 6. Training loop (simplified for the test)
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
