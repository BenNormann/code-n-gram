"""
Complete pipeline for code completion model.
Handles data preparation, model selection, and evaluation.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

from model.data_handling import (
    ensure_directory_exists,
    load_and_tokenize_data,
    split_data
)
from model.training import NGramModel, save_model
from model.evaluation import evaluate_model, print_metrics, save_metrics


def filter_comparison_metrics(metrics: Dict) -> Dict:
    """
    Filter metrics to only include essential information for model comparison.
    
    @input metrics: Full metrics dictionary
    @return: Filtered metrics with only essential information
    """
    return {
        'perplexity': metrics['perplexity'],
        'perplexity_no_punctuation': metrics['perplexity_no_punctuation'],
        'accuracy': metrics['accuracy'],
        'vocabulary_size': metrics['vocabulary_size']
    }


def select_best_model(
    train_data: List[List[str]], 
    val_data: List[List[str]], 
    n_range: Tuple[int, int] = (2, 8), 
    smoothing_k: float = 0.01,
    output_dir: str = './results'
) -> Tuple[NGramModel, Dict, Dict]:
    """
    Train and evaluate models with different n values to select the best one.
    
    @input train_data: List of tokenized methods for training
    @input val_data: List of tokenized methods for validation
    @input n_range: Tuple of (min_n, max_n) to try
    @input smoothing_k: Smoothing parameter
    @input output_dir: Directory to save metrics
    @return: Tuple of (best_model, best_metrics, all_metrics)
    """
    print("\nPerforming model selection...")
    metrics_dict = {}
    best_perplexity = float('inf')
    best_n = n_range[0]
    best_model = None
    best_metrics = None
    
    for n in range(n_range[0], n_range[1]):  # min_n to max_n-1 inclusive
        print(f"\nTraining model with n={n}")
        model = NGramModel(n=n, smoothing_k=smoothing_k)
        model.train(train_data)
        
        print(f"\nEvaluating model with n={n}...")
        metrics = evaluate_model(model, val_data)
        print_metrics(metrics)
        
        # Store metrics for comparison
        metrics_dict[f'n={n}'] = filter_comparison_metrics(metrics)
        
        # Track best model based on validation perplexity
        if metrics['perplexity'] < best_perplexity:
            best_perplexity = metrics['perplexity']
            best_n = n
            best_model = model
            best_metrics = metrics
    
    # Save all metrics to metrics_compare.json
    metrics_compare_file = os.path.join(output_dir, 'metrics_compare.json')
    with open(metrics_compare_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Comparison metrics saved to {metrics_compare_file}")
    
    print(f"\nBest model: n={best_n} with validation perplexity={best_perplexity:.2f}")
    return best_model, best_metrics, metrics_dict


def main():
    """Run complete training pipeline with model selection."""
    parser = argparse.ArgumentParser(description='Complete pipeline for code completion model')
    parser.add_argument('--processed_file', type=str, default='./data/processed_methods.csv',
                      help='Path to processed methods CSV file with "Method Code No Comments" column')
    parser.add_argument('--output_dir', type=str, default='./results',
                      help='Directory for results')
    parser.add_argument('--min_n', type=int, default=2,
                      help='Minimum n-gram size to try')
    parser.add_argument('--max_n', type=int, default=8,
                      help='Maximum n-gram size to try (exclusive)')
    parser.add_argument('--smoothing_k', type=float, default=0.01,
                      help='Smoothing parameter')
    parser.add_argument('--training_txt', type=str, default=None,
                      help='Path to a text file with one method per line to use for training')
    args = parser.parse_args()
    
    try:
        # Create output directory
        ensure_directory_exists(args.output_dir)
        
        # Load data - either from training.txt or from processed CSV file
        if args.training_txt:
            print(f"\nLoading data directly from {args.training_txt}...")
            with open(args.training_txt, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Process the lines - one method per line
            tokenized_methods = []
            for line in lines:
                line = line.strip()
                if line:
                    # Split the line into tokens
                    tokens = line.split()
                    tokenized_methods.append(tokens)
            
            print(f"Loaded {len(tokenized_methods)} methods from {args.training_txt}")
        else:
            # Load from processed CSV file
            processed_file = args.processed_file
            if not os.path.exists(processed_file):
                raise FileNotFoundError(f"Processed methods file not found: {processed_file}")
                
            print(f"\nLoading and tokenizing data from {processed_file}...")
            tokenized_methods = load_and_tokenize_data(processed_file, progress_bar=True)
        
        if not tokenized_methods:
            print("No valid methods found for training")
            sys.exit(1)
        
        # Split the full dataset for final evaluation
        train_data, val_data, test_data = split_data(tokenized_methods)
        full_training_data = train_data + val_data  # Combine train and validation for final training
        
        print(f"\nData split: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test")
        
        # Perform model selection
        best_model, _, _ = select_best_model(
            train_data, 
            val_data, 
            n_range=(args.min_n, args.max_n),
            smoothing_k=args.smoothing_k,
            output_dir=args.output_dir
        )
        
        # Train final model on the full training data
        print(f"\nTraining final model on full dataset ({len(full_training_data)} methods)...")
        final_model = NGramModel(n=best_model.n, smoothing_k=args.smoothing_k)
        final_model.train(full_training_data)
        
        print(f"\nEvaluating final model on test set ({len(test_data)} methods)...")
        final_metrics = evaluate_model(final_model, test_data)
        print("\nFinal Test Results:")
        print_metrics(final_metrics)
        
        # Save final metrics to metrics.json
        save_metrics(final_metrics, args.output_dir)
        print(f"Final metrics saved to {os.path.join(args.output_dir, 'metrics.json')}")
        
        # Save the best model
        save_model(final_model, args.output_dir, "best_model.pkl")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 