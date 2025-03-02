"""
Evaluation metrics for n-gram code completion model.
Calculates accuracy, perplexity, and other essential performance metrics.
"""

import json
import os
import math
from typing import Dict, List, Any
from tqdm import tqdm

from model.data_handling import pad_tokens


def evaluate_model(model: Any, eval_methods: List[List[str]]) -> Dict:
    """
    Evaluate model performance on evaluation methods.
    
    @input model: The trained NGramModel to evaluate
    @input eval_methods: List of tokenized methods for evaluation
    @return: Dictionary with evaluation metrics
    """
    correct = 0
    total = 0
    log_prob = 0
    
    # Track separate metrics for non-punctuation tokens
    non_punct_log_prob = 0
    non_punct_total = 0
    non_punct_correct = 0
    
    # Common punctuation for classification
    punctuation = {'{', '}', '(', ')', '[', ']', ';', ',', '.', ':', '=', '+', '-', '*', '/', '%', 
                  '&', '|', '^', '!', '~', '<', '>', '?', '@'}
    
    print("\nEvaluating on methods...")
    for tokens in tqdm(eval_methods, desc="Evaluating"):
        if not tokens:
            continue
            
        padded = pad_tokens(tokens, model.n)
        
        # Process each position in the method
        for i in range(len(padded) - model.n + 1):
            context = tuple(padded[i:i + model.n - 1])
            actual = padded[i + model.n - 1]
            
            # Determine if token is punctuation
            is_punct = actual in punctuation
            
            # OPTIMIZATION: Get both prediction and probability in one pass
            # First check if we have this exact context for prediction
            pred_token = None
            if context in model.ngrams:
                # Get the most frequent token for this context
                pred_token = max(model.ngrams[context].items(), key=lambda x: x[1])[0]
            
            # Get probability of actual token for perplexity calculation
            actual_prob = model.get_probability(context, actual)
            
            # Ensure probability is valid
            if actual_prob <= 0:
                actual_prob = model.smoothing_k  # Use smoothing parameter as minimum probability
            
            # Update log probability
            current_log_prob = math.log2(actual_prob)
            log_prob += current_log_prob
            total += 1
            
            # Track non-punctuation statistics separately
            if not is_punct:
                non_punct_log_prob += current_log_prob
                non_punct_total += 1
            
            # Check prediction accuracy
            if pred_token:
                is_correct = (actual == pred_token)
                correct += is_correct
                
                # Track non-punctuation accuracy separately
                if not is_punct and is_correct:
                    non_punct_correct += 1
            
    # Calculate final metrics
    perplexity = 2 ** (-log_prob / total) if total else float('inf')
    non_punct_perplexity = 2 ** (-non_punct_log_prob / non_punct_total) if non_punct_total else float('inf')
    
    return {
        'accuracy': correct / total if total else 0,
        'perplexity': perplexity,
        'perplexity_no_punctuation': non_punct_perplexity,
        'vocabulary_size': len(model.vocab)
    }


def save_metrics(metrics: Dict, output_dir: str):
    """
    Save evaluation metrics to JSON file.
    
    @input metrics: Dictionary of evaluation metrics
    @input output_dir: Directory to save metrics in
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, 'metrics.json')
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)


def print_metrics(metrics: Dict):
    """
    Display evaluation metrics in a readable format.
    
    @input metrics: Dictionary of evaluation metrics
    """
    print("\nPerformance Metrics:")
    print(f"Perplexity: {metrics['perplexity']:.3f}")
    print(f"Perplexity (excluding punctuation): {metrics['perplexity_no_punctuation']:.3f}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Vocabulary size: {metrics['vocabulary_size']}")
    print("-" * 50) 