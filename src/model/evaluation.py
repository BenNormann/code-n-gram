import numpy as np
from typing import Dict, List
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import math

def evaluate_model(model, test_methods: List[str]) -> Dict:
    """
    Evaluate n-gram model performance.
    
    Args:
        model: Trained NGramModel instance
        test_methods: List of test method strings
        
    Returns:
        Dict containing evaluation metrics
    """
    correct_predictions = 0
    total_predictions = 0
    total_log_prob = 0
    
    print("\nEvaluating on test methods...")
    for method in tqdm(test_methods, desc="Evaluating"):
        tokens = model.tokenize_code(method)
        padded_tokens = model.pad_tokens(tokens)
        
        for i in range(len(padded_tokens) - model.n + 1):
            context = tuple(padded_tokens[i:i + model.n - 1])
            actual = padded_tokens[i + model.n - 1]
            prediction = model.predict_next(context)
            
            if prediction:
                pred_token, prob = prediction
                if actual == pred_token:
                    correct_predictions += 1
                total_predictions += 1
                
                if prob > 0:
                    total_log_prob += math.log2(prob)
                else:
                    total_log_prob += math.log2(model.smoothing_k)

    # Calculate metrics
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    perplexity = 2 ** (-total_log_prob / total_predictions) if total_predictions > 0 else float('inf')
    
    return {
        'accuracy': accuracy,
        'perplexity': perplexity,
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'vocabulary_size': len(model.vocab)
    }

def save_metrics(metrics: Dict, output_dir: str, overwrite: bool = True):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        output_dir: Directory to save metrics
        overwrite: Whether to overwrite existing metrics file (default: True)
    """
    metrics_file = os.path.join(output_dir, 'metrics.json')
    
    if not overwrite and os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            existing_metrics = json.load(f)
        existing_metrics.update(metrics)
        metrics = existing_metrics
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

def print_metrics(metrics: Dict):
    """
    Print evaluation metrics to console.
    
    Args:
        metrics: Dictionary containing evaluation metrics
    """
    print("\nResults:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Perplexity: {metrics['perplexity']:.3f}")
    print(f"Vocabulary size: {metrics['vocabulary_size']}")
    print(f"Total predictions: {metrics['total_predictions']}")
    print(f"Correct predictions: {metrics['correct_predictions']}") 