import torch

def calculate_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=-1)
    # Assuming labels are not padded or padding is handled elsewhere for accuracy calculation
    correct_predictions = (predictions == labels).float()
    accuracy = correct_predictions.mean()
    return accuracy.item()

def calculate_perplexity(loss):
    return torch.exp(loss).item()
