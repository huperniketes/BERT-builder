from transformers import BertConfig, BertForMaskedLM

def create_cbert_model(vocab_size):
    """
    Creates a C-BERT model with a BERT-base architecture.

    Args:
        vocab_size (int): The size of the vocabulary for the model.

    Returns:
        A BertForMaskedLM model instance.
    """
    # BERT-base configuration
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2, # As per standard BERT
    )

    model = BertForMaskedLM(config=config)
    return model

if __name__ == '__main__':
    # Example of creating a model for the Char tokenizer vocab size
    # (assuming a simple ASCII vocab + special tokens)
    char_vocab_size = 128 
    model = create_cbert_model(char_vocab_size)
    
    print("C-BERT Model Configuration:")
    print(model.config)
    
    num_params = model.num_parameters()
    print(f"\nNumber of parameters: {num_params / 1_000_000:.2f}M")
