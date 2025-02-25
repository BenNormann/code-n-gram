import os
import argparse
from data_processing.mining import extract_methods_to_csv
from data_processing.preprocessing_java_methods import preprocess_methods
from model.training import main as train_model
import json

def ensure_data_exists(data_dir: str, repo_list: str) -> bool:
    """
    Ensure all necessary data files exist, running mining and preprocessing if needed.
    
    Args:
        data_dir: Directory for data files
        repo_list: Path to CSV containing repository list
    """
    os.makedirs(data_dir, exist_ok=True)
    
    extracted_methods = os.path.join(data_dir, "extracted_methods.csv")
    processed_methods = os.path.join(data_dir, "processed_methods.csv")
    
    # Check if we need to mine methods
    if not os.path.exists(extracted_methods):
        print("\nMining methods from repositories...")
        if not os.path.exists(repo_list):
            raise FileNotFoundError(f"Repository list not found: {repo_list}")
        extract_methods_to_csv(repo_list, extracted_methods, single_repo=False)
    
    # Check if we need to preprocess methods
    if not os.path.exists(processed_methods):
        print("\nPreprocessing extracted methods...")
        preprocess_methods(extracted_methods, processed_methods, language="java")

def run_training_for_n(n: int, data_dir: str, output_dir: str, metrics_dict: dict):
    """Run training for a specific n value and update metrics dictionary."""
    print(f"\nTraining model with n={n}")
    import sys
    sys.argv = [
        'training.py',
        '--data', os.path.join(data_dir, "processed_methods.csv"),
        '--output_dir', output_dir,
        '--n', str(n),
        '--overwrite_metrics', 'False'  # Don't overwrite, we'll handle it here
    ]
    
    metrics_file = os.path.join(output_dir, 'metrics.json')
    train_model()
    
    # Read the metrics for this n
    with open(metrics_file, 'r') as f:
        n_metrics = json.load(f)
    
    # Store metrics under this n value
    metrics_dict[f'n={n}'] = {
        'accuracy': n_metrics['accuracy'],
        'perplexity': n_metrics['perplexity'],
        'vocabulary_size': n_metrics['vocabulary_size'],
        'total_predictions': n_metrics['total_predictions'],
        'correct_predictions': n_metrics['correct_predictions']
    }

def main():
    parser = argparse.ArgumentParser(description='Complete pipeline for code completion model')
    parser.add_argument('--repo_list', type=str, default='./data/data.csv',
                      help='Path to CSV containing repository list')
    parser.add_argument('--data_dir', type=str, default='./data',
                      help='Directory for data files')
    parser.add_argument('--output_dir', type=str, default='./results',
                      help='Directory for results')
    args = parser.parse_args()
    
    try:
        # Step 1: Ensure data pipeline is complete
        ensure_data_exists(args.data_dir, args.repo_list)
        
        # Step 2: Train and evaluate models for n=2 to n=7
        metrics_dict = {}
        
        for n in range(2, 8):  # 2 to 7 inclusive
            run_training_for_n(n, args.data_dir, args.output_dir, metrics_dict)
        
        # Save combined metrics
        metrics_file = os.path.join(args.output_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print("\nFinal combined metrics saved to metrics.json")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 