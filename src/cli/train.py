import argparse
from cbert import trainer

def main():
    parser = argparse.ArgumentParser(description="Train a C-BERT model.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to the pre-processed training data.")
    parser.add_argument("--config", type=str, required=True, help="Path to a model configuration JSON file (e.g., BERT-base).")
    parser.add_argument("--tokenizer", type=str, required=True, choices=['char', 'keychar', 'spe'], help="Tokenizer to use.")
    parser.add_argument("--masking", type=str, default='mlm', choices=['mlm', 'wwm'], help="Masking strategy.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save checkpoints and the final model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-GPU batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Optimizer learning rate.")
    parser.add_argument("--max-steps", type=int, default=-1, help="Total number of training steps. Overrides epochs.")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from.")
    
    args = parser.parse_args()

    trainer.run(args)

if __name__ == "__main__":
    main()
