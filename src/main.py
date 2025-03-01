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
        
        metrics_dict[f'n={n}'] = metrics
        
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
    
    # Save best model metrics to metrics_selection.json
    metrics_selection_file = os.path.join(output_dir, 'metrics_selection.json')
    with open(metrics_selection_file, 'w') as f:
        json.dump({f'n={best_n}': best_metrics}, f, indent=2)
    print(f"Selection metrics saved to {metrics_selection_file}")
    
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
                      help='Use full dataset instead of evaluation subset')
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
        
        # Split data with optional sampling
        sample_size = None if args.full else 500
        train_data, val_data, test_data = split_data(tokenized_methods, sample_size=sample_size)
        
        print(f"\nData split: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test")
        
        # Perform model selection
        best_model, _, _ = select_best_model(
            train_data, 
            val_data, 
            n_range=(args.min_n, args.max_n),
            smoothing_k=args.smoothing_k,
            output_dir=args.output_dir
        )
        
        # Train final model on combined train+validation data and evaluate on test set
        print("\nTraining final model on full training data...")
        train_val_data = train_data + val_data
        final_model = NGramModel(n=best_model.n, smoothing_k=args.smoothing_k)
        final_model.train(train_val_data)
        
        print("\nEvaluating final model on test set...")
        final_metrics = evaluate_model(final_model, test_data)
        print("\nFinal Test Results:")
        print_metrics(final_metrics)
        
        # Save final metrics to metrics_final.json instead of metrics.json
        metrics_final_file = os.path.join(args.output_dir, 'metrics_final.json')
        with open(metrics_final_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        print(f"Final metrics saved to {metrics_final_file}")
        
        # Save the best model
        save_model(final_model, args.output_dir, "best_model.pkl")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 