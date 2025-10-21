import argparse
import json
import os
import torch
from transformers import BertForMaskedLM, BertConfig

from cbert.trainer import get_tokenizer # Re-use get_tokenizer
from cbert.model import create_cbert_model # Import create_cbert_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Get embeddings for a C code snippet.")
    parser.add_argument("--code", type=str, required=True, help="The C code snippet to embed.")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to the trained model directory (Hugging Face format). This directory should contain config.json, model.safetensors/pytorch_model.bin, and tokenizer files.")
    parser.add_argument("--output-file", type=str, default=None, help="File to save the JSON embeddings (default: stdout).")
    parser.add_argument("--tokenizer-type", type=str, required=True, choices=['char', 'keychar', 'spe'], help="Type of tokenizer used during training.")
    parser.add_argument("--vocab-file", type=str, default=None, help="Path to the vocabulary file (required for SentencePiece tokenizer). If not provided, attempts to load from model-dir.")
    parser.add_argument("--spm-model-file", type=str, default=None, help="Path to the SentencePiece model file (required for SentencePiece tokenizer). If not provided, attempts to load from model-dir.")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum sequence length for tokenization.")

    args = parser.parse_args()

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
    model = create_cbert_model(config) # Use create_cbert_model
    model.load_state_dict(BertForMaskedLM.from_pretrained(args.model_dir, config=config).state_dict()) # Load weights
    model.to(device)
    model.eval() # Set model to evaluation mode
    logger.info("Model loaded and set to evaluation mode.")

    # 4. Process C code snippet
    encoded_input = tokenizer.encode_plus(
        args.code,
        add_special_tokens=True,
        max_length=args.max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)

    # 5. Generate Embeddings
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Get the last hidden state of the [CLS] token (first token)
        # This is a common way to get a sentence embedding in BERT-like models
        embedding = outputs.hidden_states[-1][:, 0, :].squeeze().cpu().numpy().tolist()

    # 6. Save Results
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(embedding, f, indent=2)
        logger.info(f"Embeddings saved to {args.output_file}")
    else:
        print(json.dumps(embedding, indent=2))

if __name__ == "__main__":
    main()
