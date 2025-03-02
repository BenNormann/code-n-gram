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
    validate_data_file,
    ensure_directory_exists,
    load_and_tokenize_data,
    split_data
)
from model.training import NGramModel, train_and_evaluate, save_model
from model.evaluation import evaluate_model, print_metrics, save_metrics
from data_processing.mining import extract_methods_to_csv
from data_processing.preprocessing_java_methods import preprocess_methods


def ensure_data_exists(data_dir: str, repo_list: str):
    """
    Prepare data files, running mining and preprocessing if needed.
    
    @input data_dir: Directory for data files
    @input repo_list: Path to CSV containing repository list
    """
    os.makedirs(data_dir, exist_ok=True)
    
    extracted = os.path.join(data_dir, "extracted_methods.csv")
    processed = os.path.join(data_dir, "processed_methods.csv")
    
    # Extract methods if needed
    if not os.path.exists(extracted):
        print("\nMining methods from repositories...")
        if not os.path.exists(repo_list):
            raise FileNotFoundError(f"Repository list not found: {repo_list}")
        extract_methods_to_csv(repo_list, extracted, single_repo=False)
    
    # Preprocess methods if needed
    if not os.path.exists(processed):
        print("\nPreprocessing extracted methods...")
        preprocess_methods(extracted, processed, language="java")


def filter_comparison_metrics(metrics: Dict) -> Dict:
    """
    Filter metrics to only include essential information for model comparison.
    
    @input metrics: Full metrics dictionary
    @return: Filtered metrics with only essential information
    """
    return {
        'perplexity': metrics['perplexity'],
        'perplexity_no_padding': metrics.get('perplexity_no_padding', None),
        'accuracy': metrics['accuracy'],
        'token_type_perplexities': metrics.get('token_type_perplexities', {}),
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
        model, metrics = train_and_evaluate(train_data, val_data, n=n, smoothing_k=smoothing_k)
        
        # Store only essential metrics for comparison
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
    parser.add_argument('--repo_list', type=str, default='./data/data.csv',
                      help='Path to CSV containing repository list')
    parser.add_argument('--data_dir', type=str, default='./data',
                      help='Directory for data files')
    parser.add_argument('--output_dir', type=str, default='./results',
                      help='Directory for results')
    parser.add_argument('--full', action='store_true',
                      help='Use full dataset for model selection instead of a subset')
    parser.add_argument('--min_n', type=int, default=2,
                      help='Minimum n-gram size to try')
    parser.add_argument('--max_n', type=int, default=8,
                      help='Maximum n-gram size to try (exclusive)')
    parser.add_argument('--smoothing_k', type=float, default=0.01,
                      help='Smoothing parameter')
    args = parser.parse_args()
    
    try:
        # Prepare data pipeline
        ensure_data_exists(args.data_dir, args.repo_list)
        ensure_directory_exists(args.output_dir)
        
        # Load and tokenize data
        data_file = os.path.join(args.data_dir, "processed_methods.csv")
        print("\nLoading and tokenizing data...")
        tokenized_methods = load_and_tokenize_data(data_file, progress_bar=True)
        
        if not tokenized_methods:
            print("No valid methods found for training")
            sys.exit(1)
        
        # Load the full dataset first
        print(f"\nLoaded {len(tokenized_methods)} methods from dataset")
        
        # Split the full dataset for final evaluation
        train_full, val_full, test_data = split_data(tokenized_methods)
        full_training_data = train_full + val_full  # Combine train and validation for final training
        
        # For model selection, use either the full dataset or a sample
        if args.full:
            # Use the full training and validation sets for model selection
            train_selection = train_full
            val_selection = val_full
            print(f"Using full dataset for model selection: {len(train_selection)} train, {len(val_selection)} validation")
        else:
            # Use a smaller sample for faster model selection
            sample_size = 500
            train_sample, val_sample, _ = split_data(tokenized_methods, sample_size=sample_size)
            train_selection = train_sample
            val_selection = val_sample
            print(f"Using sample for model selection: {len(train_selection)} train, {len(val_selection)} validation")
        
        # Perform model selection on the selection dataset
        best_model, _, _ = select_best_model(
            train_selection, 
            val_selection, 
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
        metrics_file = os.path.join(args.output_dir, 'metrics.json')
        
        # Save both full metrics and filtered metrics
        final_metrics_to_save = {
            'full': final_metrics,
            'essential': filter_comparison_metrics(final_metrics)
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(final_metrics_to_save, f, indent=2)
        print(f"Final metrics saved to {metrics_file}")
        
        # Save the best model
        save_model(final_model, args.output_dir, "best_model.pkl")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 