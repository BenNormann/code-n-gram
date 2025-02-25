import os
import argparse
from data_processing.mining import extract_methods_to_csv
from data_processing.preprocessing_java_methods import preprocess_methods
from model.training import main as train_model

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
        
        # Step 2: Train and evaluate models for n=2 through n=7
        for n in range(2, 8):  # 2 through 7 inclusive
            print(f"\nTraining model with n-gram size {n}...")
            # Pass the n value and output directory to train_model
            # Assuming train_model accepts these parameters
            train_model(n=n, output_dir=os.path.join(args.output_dir, f"model_n{n}"))
            print(f"Completed training for n={n}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
