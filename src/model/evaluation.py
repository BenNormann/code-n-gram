"""
Evaluation metrics for n-gram code completion model.
Calculates accuracy, perplexity, and other performance metrics.
"""

import json
import os
import math
from typing import Dict, List, Any
from tqdm import tqdm
import random

from model.data_handling import pad_tokens


def evaluate_model(model: Any, eval_methods: List[List[str]], max_tokens: int = 10000) -> Dict:
    """
    Evaluate model performance on evaluation methods.
    
    @input model: The trained NGramModel to evaluate
    @input eval_methods: List of tokenized methods for evaluation
    @input max_tokens: Maximum number of tokens to evaluate (for performance)
    @return: Dictionary with evaluation metrics
    """
    correct = 0
    total = 0
    log_prob = 0
    tokens_processed = 0
    
    # Shuffle methods to get a representative sample if we hit max_tokens
    random.shuffle(eval_methods)
    
    print("\nEvaluating on methods...")
    for tokens in tqdm(eval_methods, desc="Evaluating"):
        if not tokens:
            continue
            
        padded = pad_tokens(tokens, model.n)
        
        # Process each position in the method
        for i in range(len(padded) - model.n + 1):
            # Check if we've processed enough tokens
            if tokens_processed >= max_tokens:
                break
                
            context = tuple(padded[i:i + model.n - 1])
            actual = padded[i + model.n - 1]
            
            # Get probability of actual token for perplexity calculation
            actual_prob = model.get_probability(context, actual)
            log_prob += math.log2(actual_prob) if actual_prob > 0 else math.log2(model.smoothing_k)
            
            # Get prediction for accuracy calculation
            if pred := model.predict_next(context):
                pred_token, _ = pred
                correct += (actual == pred_token)
                total += 1
            
            tokens_processed += 1
        
        # Check if we've processed enough tokens
        if tokens_processed >= max_tokens:
            print(f"\nReached maximum token limit ({max_tokens}). Stopping evaluation.")
            break
    
    # Calculate final metrics
    return {
        'accuracy': correct / total if total else 0,
        'perplexity': 2 ** (-log_prob / total) if total else float('inf'),
        'total_predictions': total,
        'correct_predictions': correct,
        'vocabulary_size': len(model.vocab),
        'tokens_evaluated': tokens_processed
    }


def save_metrics(metrics: Dict, output_dir: str, overwrite: bool = True):
    """
    Save evaluation metrics to JSON file.
    
    @input metrics: Dictionary of evaluation metrics
    @input output_dir: Directory to save metrics in
    @input overwrite: Whether to overwrite existing metrics file
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, 'metrics.json')
    
    if not overwrite and os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = {**json.load(f), **metrics}
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)


def print_metrics(metrics: Dict):
    """
    Display evaluation metrics in a readable format.
    
    @input metrics: Dictionary of evaluation metrics
    """
    print("\nResults:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Perplexity: {metrics['perplexity']:.3f}")
    print(f"Vocabulary size: {metrics['vocabulary_size']}")
    print(f"Total predictions: {metrics['total_predictions']}")
    print(f"Correct predictions: {metrics['correct_predictions']}")
    if 'tokens_evaluated' in metrics:
        print(f"Tokens evaluated: {metrics['tokens_evaluated']}") 