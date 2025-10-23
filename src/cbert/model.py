from transformers import BertConfig, BertForMaskedLM

def create_cbert_model(config: BertConfig) -> BertForMaskedLM:
    """
    Creates a C-BERT model with a given BertConfig.

    Args:
        config (BertConfig): The configuration object for the model.

    Returns:
        A BertForMaskedLM model instance.
        
    Raises:
        ValueError: If config is None or has invalid parameters.
        TypeError: If config is not a BertConfig instance.
    """
    if config is None:
        raise ValueError("Config cannot be None")
    
    if not isinstance(config, BertConfig):
        raise TypeError(f"Expected BertConfig, got {type(config)}")
    
    # Validate essential config parameters
    if config.vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {config.vocab_size}")
    if config.hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {config.hidden_size}")
    if config.num_hidden_layers <= 0:
        raise ValueError(f"num_hidden_layers must be positive, got {config.num_hidden_layers}")
    
    model = BertForMaskedLM(config=config)
    return model

def get_model_info(model: BertForMaskedLM) -> dict:
    """
    Get comprehensive information about a C-BERT model.
    
    Args:
        model: The BERT model instance.
        
    Returns:
        Dictionary containing model information.
    """
    if model is None:
        raise ValueError("Model cannot be None")
        
    config = model.config
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.max_position_embeddings,
        "total_parameters": num_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": num_params * 4 / (1024 * 1024),
    }

if __name__ == '__main__':
    # Example: creating a Character-level tokenizer model
    # (assuming a simple ASCII vocab + special tokens)
    # In a real scenario, config would be loaded from a file.
    example_config = BertConfig(
        vocab_size=128,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
    )
    model = create_cbert_model(example_config)
    
    info = get_model_info(model)
    
    print("C-BERT Model Information:")
    print(f"Parameters: {info['total_parameters'] / 1_000_000:.2f}M")
    print(f"Model size: {info['model_size_mb']:.1f} MB")
    print(f"Vocab size: {info['vocab_size']:,}")
    print(f"Hidden size: {info['hidden_size']}")
    print(f"Layers: {info['num_hidden_layers']}")
    print(f"Attention heads: {info['num_attention_heads']}")
