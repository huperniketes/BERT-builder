import os
import json
import torch
import logging
import random
import re
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizer

from .model import create_cbert_model
from .tokenizer import CharTokenizer, KeyCharTokenizer, SentencePieceTokenizer
from .validation import ValidationPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, max_length: int, masking_strategy: str = 'mlm', dataset_lineoffsets_file: str = None):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.masking_strategy = masking_strategy
        
        # For WWM, we need to identify whole words. This is a simple regex for C-like identifiers.
        self.word_regex = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)')

        # Try to load existing dataset lineoffsets
        if dataset_lineoffsets_file and os.path.exists(dataset_lineoffsets_file):
            logger.info(f"Loading dataset lineoffsets from {dataset_lineoffsets_file}")
            with open(dataset_lineoffsets_file, 'r') as f:
                state = json.load(f)
                self.line_offsets = state['line_offsets']
        else:
            # Build line offsets from scratch
            self.line_offsets = []
            current_offset = 0
            num_meaningful_lines = 0

            # Open the file once and keep it open
            temp_file = open(file_path, 'r', encoding='utf-8')

            # Reset file pointer to the beginning before iterating
            temp_file.seek(0)
            for line_num, line in enumerate(temp_file):
                stripped_line = line.strip()
                if not stripped_line: # Skip empty lines
                    current_offset += len(line.encode('utf-8'))
                    continue
                
                encoded_content = self.tokenizer.encode(stripped_line, add_special_tokens=False, max_length=self.max_length, truncation=True)
                if encoded_content:
                    unk_token_id = self.tokenizer.unk_token_id
                    if unk_token_id is not None:
                        unk_count = encoded_content.count(unk_token_id)
                        if len(encoded_content) > 0:
                            unk_ratio = unk_count / len(encoded_content)
                            if unk_ratio > 0.5:
                                logger.warning(f"High UNK token ratio ({unk_ratio:.2f}) in line {line_num} of {file_path}: '{stripped_line[:100]}...'")

                    self.line_offsets.append(current_offset)
                    num_meaningful_lines += 1
                current_offset += len(line.encode('utf-8'))
            
            temp_file.close()
            logger.info(f"Loaded {num_meaningful_lines} meaningful lines from {file_path}")
            
            # Save dataset lineoffsets if requested
            if dataset_lineoffsets_file:
                self.save_lineoffsets(dataset_lineoffsets_file)
        
        # Open the file for reading during training
        self.file = open(file_path, 'r', encoding='utf-8')

    def __del__(self):
        if hasattr(self, 'file') and self.file:
            self.file.close()

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, idx):
        offset = self.line_offsets[idx]
        self.file.seek(offset)
        line = self.file.readline().strip()
        
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
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
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
        
        # Get words from the text using the regex
        words_in_text = [(m.group(0), m.start(), m.end()) for m in self.word_regex.finditer(text)]
        
        if not words_in_text:
            return self.mask_tokens_mlm(inputs, mlm_probability) # Fallback to MLM if no words are found

        # Decide which words to mask
        words_to_mask_spans = []
        for word, start_char, end_char in words_in_text:
            if random.random() < mlm_probability:
                words_to_mask_spans.append((start_char, end_char))
        
        if not words_to_mask_spans:
            return inputs, labels # Nothing to mask

        # Get token spans from the tokenizer
        token_spans = self.tokenizer._get_token_spans(text)
        
        # Identify tokens to mask based on word spans
        masked_indices = torch.zeros_like(inputs).bool()
        for token_idx, (token, token_start_char, token_end_char) in enumerate(token_spans):
            for word_start_char, word_end_char in words_to_mask_spans:
                # Check for overlap between token span and word span
                if max(token_start_char, word_start_char) < min(token_end_char, word_end_char):
                    if inputs[token_idx] not in self.tokenizer.all_special_ids:
                        masked_indices[token_idx] = True
                        break # Move to the next token once it's marked for masking

        labels[~masked_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token id
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels
    
    def save_lineoffsets(self, lineoffsets_file: str):
        """Save dataset lineoffsets to avoid re-initialization."""
        state = {'line_offsets': self.line_offsets}
        with open(lineoffsets_file, 'w') as f:
            json.dump(state, f)


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
    
    # Initialize validation pipeline
    validator = ValidationPipeline(os.path.join(args.output_dir, "validation"))

    # 1. Load Config
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    config = BertConfig(**config_dict)

    # 2. Get Tokenizer
    tokenizer = get_tokenizer(args.tokenizer, args.vocab_file, args.spm_model_file)
    config.vocab_size = len(tokenizer)

    # 3. Create Model
    model = create_cbert_model(config)
    model.resize_token_embeddings(len(tokenizer))


    # 4. Create Dataset and DataLoader
    # Try to use lineoffsets file from data preprocessing first
    dataset_lineoffsets_file = args.dataset_dir.replace('.txt', '_lineoffsets.json')
    if not os.path.exists(dataset_lineoffsets_file):
        # Fallback to generating in output dir
        dataset_lineoffsets_file = os.path.join(args.output_dir, 'dataset_lineoffsets.json')
    dataset = TextDataset(args.dataset_dir, tokenizer, config.max_position_embeddings, args.masking, dataset_lineoffsets_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Validate tokenization with sample data
    sample_texts = []
    try:
        with open(args.dataset_dir, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 10:  # Sample first 10 lines
                    break
                if line.strip():
                    sample_texts.append(line.strip())
    except Exception as e:
        logger.warning(f"Could not sample texts for validation: {e}")
    
    if sample_texts:
        validator.validate_tokenization(tokenizer, sample_texts)

    # 5. Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # 6. Resumption
    start_step = 0
    if args.resume_from_checkpoint:
        checkpoint_file = os.path.join(args.resume_from_checkpoint, 'training_state.bin')
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        if not os.access(checkpoint_file, os.R_OK):
            raise PermissionError(f"Cannot read checkpoint file: {checkpoint_file}")
        
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(checkpoint_file)
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

            # Validate training batch
            batch_validation = validator.validate_training_batch(batch, outputs)
            if not batch_validation.passed:
                logger.error(f"Training batch validation failed at step {global_step}")

            # Checkpointing
            if global_step > 0 and global_step % 1 == 0: # Checkpoint every step
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
                
                # Validate checkpoint
                validator.validate_model_checkpoint(model, tokenizer, checkpoint_dir)

            global_step += 1
        
        if hasattr(args, 'max_steps') and args.max_steps > 0 and global_step >= args.max_steps:
            break


    # 8. Save final model
    logger.info(f"Saving final model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Final validation summary
    summary = validator.get_summary()
    logger.info(f"Validation Summary: {summary['passed_stages']}/{summary['total_stages']} stages passed")
    if summary['failed_stages']:
        logger.warning(f"Failed stages: {summary['failed_stages']}")
    
    logger.info("Training run finished.")
