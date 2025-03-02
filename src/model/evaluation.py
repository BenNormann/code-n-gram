"""
Evaluation metrics for n-gram code completion model.
Calculates accuracy, perplexity, and other performance metrics.
"""

import json
import os
import math
import statistics
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import random

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
    
    # Track separate metrics for non-padding tokens
    non_padding_correct = 0
    non_padding_total = 0
    non_padding_log_prob = 0
    
    # Track token type statistics
    token_type_stats = {
        'padding': {'count': 0, 'correct': 0, 'log_prob_sum': 0},
        'punctuation': {'count': 0, 'correct': 0, 'log_prob_sum': 0},
        'keywords': {'count': 0, 'correct': 0, 'log_prob_sum': 0},
        'identifiers': {'count': 0, 'correct': 0, 'log_prob_sum': 0},
        'other': {'count': 0, 'correct': 0, 'log_prob_sum': 0}
    }
    
    # Common Java keywords and punctuation for classification
    java_keywords = {'public', 'private', 'protected', 'static', 'final', 'void', 'int', 'boolean', 
                    'String', 'class', 'interface', 'extends', 'implements', 'return', 'if', 'else', 
                    'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'new', 'try', 'catch', 
                    'throw', 'throws', 'this', 'super', 'null', 'true', 'false'}
    punctuation = {'{', '}', '(', ')', '[', ']', ';', ',', '.', ':', '=', '+', '-', '*', '/', '%', 
                  '&', '|', '^', '!', '~', '<', '>', '?', '@'}
    
    print("\nEvaluating on methods...")
    for method_idx, tokens in enumerate(tqdm(eval_methods, desc="Evaluating")):
        if not tokens:
            continue
            
        padded = pad_tokens(tokens, model.n)
        
        # Process each position in the method
        for i in range(len(padded) - model.n + 1):
            context = tuple(padded[i:i + model.n - 1])
            actual = padded[i + model.n - 1]
            
            # Determine token type
            is_padding = actual in ['<START>', '<END>'] or any(t in ['<START>', '<END>'] for t in context)
            token_type = 'padding'
            if not is_padding:
                if actual in punctuation:
                    token_type = 'punctuation'
                elif actual in java_keywords:
                    token_type = 'keywords'
                elif actual[0].isalpha() or actual[0] == '_':
                    token_type = 'identifiers'
                else:
                    token_type = 'other'
            
            # Get probability of actual token for perplexity calculation
            actual_prob = model.get_probability(context, actual)
            
            # Ensure probability is valid
            if actual_prob <= 0:
                actual_prob = model.smoothing_k  # Use smoothing parameter as minimum probability
            
            # Update log probability
            current_log_prob = math.log2(actual_prob)
            log_prob += current_log_prob
            
            # Update token type statistics
            token_type_stats[token_type]['count'] += 1
            token_type_stats[token_type]['log_prob_sum'] += current_log_prob
            
            # Track non-padding statistics separately
            if not is_padding:
                non_padding_log_prob += current_log_prob
                non_padding_total += 1
            
            # Get prediction for accuracy calculation
            if pred := model.predict_next(context):
                pred_token, _ = pred
                is_correct = (actual == pred_token)
                correct += is_correct
                total += 1
                
                # Update token type accuracy
                if is_correct:
                    token_type_stats[token_type]['correct'] += 1
                
                # Track non-padding accuracy separately
                if not is_padding:
                    if is_correct:
                        non_padding_correct += 1
    
    # Calculate final metrics
    perplexity = 2 ** (-log_prob / total) if total else float('inf')
    non_padding_perplexity = 2 ** (-non_padding_log_prob / non_padding_total) if non_padding_total else float('inf')
    
    # Calculate token type perplexities
    token_type_perplexities = {}
    for token_type, stats in token_type_stats.items():
        if stats['count'] > 0:
            token_type_perplexities[token_type] = 2 ** (-stats['log_prob_sum'] / stats['count'])
            stats['accuracy'] = stats['correct'] / stats['count'] if stats['count'] > 0 else 0
    
    return {
        'accuracy': correct / total if total else 0,
        'perplexity': perplexity,
        'perplexity_no_padding': non_padding_perplexity,
        'total_predictions': total,
        'correct_predictions': correct,
        'vocabulary_size': len(model.vocab),
        'non_padding_accuracy': non_padding_correct / non_padding_total if non_padding_total else 0,
        'non_padding_tokens': non_padding_total,
        'token_type_stats': token_type_stats,
        'token_type_perplexities': token_type_perplexities
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
    print("\nPerplexity Results:")
    print(f"Perplexity (all tokens): {metrics['perplexity']:.3f}")
    
    if 'perplexity_no_padding' in metrics:
        print(f"Perplexity (excluding padding): {metrics['perplexity_no_padding']:.3f}")
    
    if 'token_type_perplexities' in metrics:
        print("\nPerplexity by token type:")
        for token_type, perplexity in metrics['token_type_perplexities'].items():
            print(f"  {token_type}: {perplexity:.3f}")
    
    # Add a summary line for the best model selection
    if 'accuracy' in metrics:
        print(f"\nModel accuracy: {metrics['accuracy']:.3f}")
        
    # Print a separator line for readability
    print("-" * 50) 