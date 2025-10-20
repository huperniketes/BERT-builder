import argparse
import json
import torch
from transformers import BertForMaskedLM, BertConfig

def main():
    parser = argparse.ArgumentParser(description="Get embeddings for a C code snippet.")
    parser.add_argument("--code", type=str, required=True, help="The C code snippet to embed.")
    # In a real scenario, we would also have arguments for model_dir, tokenizer, etc.
    args = parser.parse_args()

    # 1. Load Model (dummy for now, similar to evaluate.py)
    # In a real scenario, we would load a trained model and its config
    # For this test, we'll create a dummy config and model
    config = BertConfig(
        vocab_size=128,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        max_position_embeddings=128
    )
    model = BertForMaskedLM(config=config)
    model.eval()

    # 2. Process C code (simplified for now)
    # In a real scenario, this would involve tokenization and feeding to the model
    # For this test, we'll return a dummy embedding
    dummy_embedding = [float(ord(c) % 100) / 100.0 for c in args.code] # Example: [0.73, 0.97, ...]
    if not dummy_embedding: # Ensure it's not empty for the test
        dummy_embedding = [0.0]

    # 3. Print embedding to stdout
    print(json.dumps(dummy_embedding))

if __name__ == "__main__":
    main()
