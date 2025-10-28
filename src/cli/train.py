import argparse
import os
import sys
from cbert import trainer

def main():
    parser = argparse.ArgumentParser(description="Train a C-BERT model.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to the pre-processed training data.")
    parser.add_argument("--config", type=str, required=True, help="Path to a model configuration JSON file (e.g., BERT-base).")
    parser.add_argument("--tokenizer", type=str, required=True, choices=['char', 'keychar', 'spe'], help="Tokenizer to use.")
    parser.add_argument("--vocab-file", type=str, default=None, help="Path to the vocabulary file (required for SentencePiece tokenizer).")
    parser.add_argument("--spm-model-file", type=str, default=None, help="Path to the SentencePiece model file (required for SentencePiece tokenizer).")
    parser.add_argument("--masking", type=str, default='mlm', choices=['mlm', 'wwm'], help="Masking strategy.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save checkpoints and the final model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-GPU batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Optimizer learning rate.")
    parser.add_argument("--max-steps", type=int, default=-1, help="Total number of training steps. Overrides epochs.")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from.")
    
    args = parser.parse_args()

    # Validate SentencePiece tokenizer requirements
    if args.tokenizer == 'spe':
        if not args.vocab_file or not args.spm_model_file:
            print("Error: --vocab-file and --spm-model-file are required when using SentencePiece tokenizer")
            sys.exit(1)

    # Validate file paths exist
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory does not exist: {args.dataset_dir}")
        sys.exit(1)
    if not os.path.exists(args.config):
        print(f"Error: Config file does not exist: {args.config}")
        sys.exit(1)
    if args.vocab_file and not os.path.exists(args.vocab_file):
        print(f"Error: Vocab file does not exist: {args.vocab_file}")
        sys.exit(1)
    if args.spm_model_file and not os.path.exists(args.spm_model_file):
        print(f"Error: SentencePiece model file does not exist: {args.spm_model_file}")
        sys.exit(1)
    if args.resume_from_checkpoint:
        checkpoint_file = os.path.join(args.resume_from_checkpoint, 'training_state.bin')
        if not os.path.exists(checkpoint_file):
            print(f"Error: Checkpoint file does not exist: {checkpoint_file}")
            sys.exit(1)
        if not os.access(checkpoint_file, os.R_OK):
            print(f"Error: Cannot read checkpoint file: {checkpoint_file}")
            sys.exit(1)

    # Validate numeric arguments
    if args.epochs <= 0:
        print("Error: --epochs must be greater than 0")
        sys.exit(1)
    if args.batch_size <= 0:
        print("Error: --batch-size must be greater than 0")
        sys.exit(1)
    if args.learning_rate <= 0:
        print("Error: --learning-rate must be greater than 0")
        sys.exit(1)

    trainer.run(args)

if __name__ == "__main__":
    main()
