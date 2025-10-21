import os
import json
import torch
import logging
import random
import re
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizer

from .tokenizer import CharTokenizer, KeyCharTokenizer, SentencePieceTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, max_length: int, masking_strategy: str = 'mlm'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.masking_strategy = masking_strategy
        
        # For WWM, we need to identify whole words. This is a simple regex for C-like identifiers.
        self.word_regex = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)')

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_lines = f.readlines()

        self.lines = []
        for line in raw_lines:
            stripped_line = line.strip()
            if not stripped_line: # Skip empty lines
                continue
            
            # Encode to check for meaningful content (non-special tokens)
            # Use add_special_tokens=False to check for actual content tokens
            encoded_content = self.tokenizer.encode(stripped_line, add_special_tokens=False, max_length=self.max_length, truncation=True)
            if encoded_content: # If there are any actual content tokens
                # Tokenization Sanity Check: Warn if too many UNK tokens
                unk_token_id = self.tokenizer.unk_token_id
                if unk_token_id is not None:
                    unk_count = encoded_content.count(unk_token_id)
                    unk_ratio = unk_count / len(encoded_content)
                    if unk_ratio > 0.5: # Threshold for warning (e.g., more than 50% UNK tokens)
                        logger.warning(f"High UNK token ratio ({unk_ratio:.2f}) in line: '{stripped_line[:100]}...'")

                self.lines.append(stripped_line)

        logger.info(f"Loaded {len(self.lines)} meaningful lines from {file_path}")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        
        # Tokenize the line
        token_ids = self.tokenizer.encode(line, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True)
        
        input_ids = torch.tensor(token_ids)
        labels = input_ids.clone()

        if self.masking_strategy == 'mlm':
            input_ids, labels = self.mask_tokens_mlm(input_ids)
        elif self.masking_strategy == 'wwm':
            input_ids, labels = self.mask_tokens_wwm(input_ids, line)
        else:
            raise ValueError(f"Unknown masking strategy: {self.masking_strategy}")

        return {"input_ids": input_ids, "labels": labels}

    def mask_tokens_mlm(self, inputs: torch.Tensor, mlm_probability=0.15):
        """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token id
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def mask_tokens_wwm(self, inputs: torch.Tensor, text: str, mlm_probability=0.15):
        """Prepare masked tokens inputs/labels for whole word masking."""
        labels = inputs.clone()
        
        words = self.word_regex.findall(text)
        if not words:
            return self.mask_tokens_mlm(inputs, mlm_probability) # Fallback to MLM if no words are found

        masked_words = []
        for word in words:
            if random.random() < mlm_probability:
                masked_words.append(word)
        
        if not masked_words:
            return inputs, labels # Nothing to mask

        # Create a regex to find all occurrences of the chosen words
        mask_regex = re.compile(r'\b(' + '|'.join(re.escape(w) for w in masked_words) + r')\b')
        
        # This is a simplified approach. A more robust implementation would work with token-level spans.
        # Here, we'll just mask all occurrences of the word in the line.
        
        # Find all token indices that correspond to the words to be masked
        masked_indices = torch.zeros_like(inputs).bool()
        for match in mask_regex.finditer(text):
            start, end = match.span()
            # This is a simplification: it doesn't perfectly align with the tokenizer.
            # A more rigorous approach would map character spans to token spans.
            for i in range(start, end):
                # This is not quite right, but it's a starting point.
                # We need a way to map character position to token position.
                # For now, we'll just mask the tokens that are not special tokens.
                if inputs[i] not in self.tokenizer.all_special_ids:
                     masked_indices[i] = True


        labels[~masked_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token id
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels


def get_tokenizer(tokenizer_name: str, vocab_file: str = None, spm_model_file: str = None) -> PreTrainedTokenizer:
    if tokenizer_name == 'char':
        tok = CharTokenizer(vocab_file=vocab_file)
        return tok
    elif tokenizer_name == 'keychar':
        tok = KeyCharTokenizer(vocab_file=vocab_file)
        return tok
    elif tokenizer_name == 'spe':
        if not vocab_file or not spm_model_file:
            raise ValueError("For SentencePieceTokenizer, both vocab_file and spm_model_file must be provided.")
        tok = SentencePieceTokenizer(vocab_file=vocab_file, spm_model_file=spm_model_file)
        return tok
    else:
        raise ValueError(f"Tokenizer '{tokenizer_name}' not supported.")


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
    tokenizer = get_tokenizer(args.tokenizer, args.vocab_file, args.spm_model_file)

    # 3. Create Model
    model = BertForMaskedLM(config=config)
    model.resize_token_embeddings(len(tokenizer))


    # 4. Create Dataset and DataLoader
    dataset = TextDataset(args.dataset_dir, tokenizer, config.max_position_embeddings, args.masking)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

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
                # Only calculate accuracy on masked tokens
                masked_tokens = batch['labels'] != -100
                correct = (predictions[masked_tokens] == batch['labels'][masked_tokens]).sum().item()
                total = masked_tokens.sum().item()
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
                tokenizer.save_pretrained(checkpoint_dir)
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
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training run finished.")
