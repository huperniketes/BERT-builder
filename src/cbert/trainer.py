import os
import json
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizer

from .tokenizer import CharTokenizer, KeyCharTokenizer, SentencePieceTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        # A real implementation would have a more sophisticated tokenization and masking strategy
        # For now, we'll stick to a simple implementation for the integration test
        if isinstance(self.tokenizer, CharTokenizer):
            token_ids = [ord(c) for c in line]
        else:
            token_ids = [ord(c) for c in line] # Fallback for now

        # Pad or truncate
        if len(token_ids) < self.max_length:
            token_ids += [0] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_length]

        input_ids = torch.tensor(token_ids)
        labels = input_ids.clone()

        # Simple masking for MLM (replace with a more robust strategy later)
        # Mask 15% of the tokens
        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < 0.15) * (input_ids != 0) # don't mask padding
        
        # Replace masked tokens with a mask token ID (e.g., 4 for '[MASK]')
        # This is a placeholder, a real implementation would get the mask_token_id from the tokenizer
        mask_token_id = 4 
        selection = torch.flatten(mask_arr.nonzero()).tolist()
        input_ids[selection] = mask_token_id

        return {"input_ids": input_ids, "labels": labels}


def get_tokenizer(tokenizer_name: str, config: BertConfig) -> PreTrainedTokenizer:
    if tokenizer_name == 'char':
        tok = CharTokenizer()
        tok.vocab_size = config.vocab_size
        return tok
    elif tokenizer_name == 'keychar':
        tok = KeyCharTokenizer()
        tok.vocab_size = config.vocab_size
        return tok
    else:
        raise ValueError(f"Tokenizer '{tokenizer_name}' not supported in this test.")


def run(args):
    """Main training function."""
    logger.info("Starting training run...")
    logger.info(f"Arguments: {args}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Config
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    config = BertConfig(**config_dict)

    # 2. Get Tokenizer
    tokenizer = get_tokenizer(args.tokenizer, config)

    # 3. Create Model
    model = BertForMaskedLM(config=config)

    # 4. Create Dataset and DataLoader
    dataset = TextDataset(args.dataset_dir, tokenizer, config.max_position_embeddings)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # 5. Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # 6. Resumption
    start_step = 0
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(os.path.join(args.resume_from_checkpoint, 'training_state.bin'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1

    # 7. Training loop
    model.train()
    global_step = start_step
    for epoch in range(args.epochs):
        for batch in dataloader:
            if hasattr(args, 'max_steps') and args.max_steps > 0 and global_step >= args.max_steps:
                break

            optimizer.zero_grad()
            
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()

            # Logging
            if global_step % 1 == 0: # Log every step
                # Calculate accuracy
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct = (predictions == batch['labels']).sum().item()
                total = (batch['labels'] != 0).sum().item() # ignore padding
                accuracy = correct / total if total > 0 else 0

                log_entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": loss.item(),
                    "accuracy": accuracy
                }
                logger.info(log_entry)
                with open(os.path.join(args.output_dir, 'training_log.json'), 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')

            # Checkpointing
            if global_step > 0 and global_step % 5 == 0: # Checkpoint every 5 steps
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                logger.info(f"Saving checkpoint to {checkpoint_dir}")
                model.save_pretrained(checkpoint_dir)
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(checkpoint_dir, 'training_state.bin'))

            global_step += 1
        
        if hasattr(args, 'max_steps') and args.max_steps > 0 and global_step >= args.max_steps:
            break


    # 8. Save final model
    logger.info(f"Saving final model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    
    logger.info("Training run finished.")
