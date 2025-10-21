from transformers import BertConfig, BertForMaskedLM

def create_cbert_model(config: BertConfig) -> BertForMaskedLM:
    """
    Creates a C-BERT model with a given BertConfig.

    Args:
        config (BertConfig): The configuration object for the model.

    Returns:
        A BertForMaskedLM model instance.
    """
    model = BertForMaskedLM(config=config)
    return model

if __name__ == '__main__':
    # Example of creating a model for the Char tokenizer vocab size
    # (assuming a simple ASCII vocab + special tokens)
    # In a real scenario, config would be loaded from a file.
    char_vocab_size = 128 
    example_config = BertConfig(
        vocab_size=char_vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
    )
    model = create_cbert_model(example_config)
    
    print("C-BERT Model Configuration:")
    print(model.config)
    
    num_params = model.num_parameters()
    print(f"\nNumber of parameters: {num_params / 1_000_000:.2f}M")
