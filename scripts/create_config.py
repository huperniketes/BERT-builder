#!/usr/bin/env python3
"""
Create C-BERT model configuration files for different tokenizer types and model sizes.
"""

import argparse
import json
import os
from transformers import BertConfig

def create_config(vocab_size, hidden_size=768, num_layers=12, num_heads=12, 
                 intermediate_size=3072, max_length=512, output_path=None):
    """Create a BertConfig and save it as JSON."""
    
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_length,
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        pad_token_id=0,
        position_embedding_type="absolute",
    )
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        config.save_pretrained(os.path.dirname(output_path))
        print(f"Config saved to {output_path}")
    
    return config

def main():
    parser = argparse.ArgumentParser(description="Create C-BERT configuration files")
    parser.add_argument("--vocab-size", type=int, required=True, help="Vocabulary size")
    parser.add_argument("--hidden-size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--output", type=str, required=True, help="Output config file path")
    
    args = parser.parse_args()
    
    # Validate that num_heads divides hidden_size
    if args.hidden_size % args.num_heads != 0:
        print(f"Error: hidden_size ({args.hidden_size}) must be divisible by num_heads ({args.num_heads})")
        return
    
    intermediate_size = args.hidden_size * 4  # Standard BERT ratio
    
    config = create_config(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=intermediate_size,
        max_length=args.max_length,
        output_path=args.output
    )
    
    print(f"Created config with {sum(p.numel() for p in config.to_dict().values() if isinstance(p, int)) / 1000000:.1f}M parameters")

if __name__ == "__main__":
    main()