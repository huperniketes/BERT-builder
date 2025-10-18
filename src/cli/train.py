import argparse
from cbert import trainer

def main():
    parser = argparse.ArgumentParser(description="Train a C-BERT model.")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True, choices=['char', 'keychar', 'spe'])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()

    # This will fail until the trainer is implemented
    trainer.run(args)

if __name__ == "__main__":
    main()
