import argparse
import os
from cbert.tokenizer import SentencePieceTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer.")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the text corpus file for training (e.g., c_corpus_cleaned.txt).")
    parser.add_argument("--model_prefix", type=str, required=True, help="Prefix for the output SentencePiece model files (e.g., cbert_spm).")
    parser.add_argument("--vocab_size", type=int, default=8000, help="Vocabulary size for the SentencePiece model.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the trained tokenizer model files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    full_model_prefix = os.path.join(args.output_dir, args.model_prefix)

    logger.info(f"Starting SentencePiece tokenizer training with corpus: {args.corpus_path}")
    logger.info(f"Output model prefix: {full_model_prefix}")
    logger.info(f"Vocabulary size: {args.vocab_size}")

    try:
        SentencePieceTokenizer.train(args.corpus_path, full_model_prefix, args.vocab_size)
        logger.info(f"SentencePiece tokenizer training complete. Model saved to {full_model_prefix}.model and {full_model_prefix}.vocab")
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during tokenizer training: {e}")

if __name__ == "__main__":
    main()
