"""
Evaluation module for C-BERT model.
Provides evaluation metrics and functions for model assessment.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging

def evaluate_model(model, dataloader, device: str = 'cpu') -> Dict[str, float]:
    """
    Evaluate model performance on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing evaluation data
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate loss (assuming MLM task)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                # Create labels by masking some tokens
                labels = input_ids.clone()
                
                # Calculate loss
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    labels.view(-1), 
                    ignore_index=0
                )
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=-1)
                mask = attention_mask.bool()
                correct = (predictions == labels) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / max(total_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'perplexity': perplexity
    }

def calculate_perplexity(model, dataloader, device: str = 'cpu') -> float:
    """Calculate perplexity metric."""
    results = evaluate_model(model, dataloader, device)
    return results['perplexity']

def calculate_accuracy(model, dataloader, device: str = 'cpu') -> float:
    """Calculate accuracy metric."""
    results = evaluate_model(model, dataloader, device)
    return results['accuracy']