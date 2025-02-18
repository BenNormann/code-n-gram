import os
from data_processing.mining import extract_methods_to_csv
from data_processing.preprocessing_java_methods import preprocess_methods

def main():
    """Main function to run the Java method mining and preprocessing pipeline."""
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Set up paths
    data_dir = os.path.join(project_root, "data")
    data_csv = os.path.join(data_dir, "data.csv")
    extracted_csv = os.path.join(data_dir, "extracted_methods.csv")
    processed_csv = os.path.join(data_dir, "processed_methods.csv")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if input file exists
    if not os.path.exists(data_csv):
        print(f"Error: Repository list not found: {data_csv}")
        return
    
    print("Starting Java method mining pipeline...")
    
    # Step 1: Mine methods from repositories
    print("\n=== Step 1: Mining Methods ===")
    extract_methods_to_csv(data_csv, extracted_csv, single_repo=False)
    
    # Step 2: Preprocess methods
    print("\n=== Step 2: Preprocessing Methods ===")
    preprocess_methods(extracted_csv, processed_csv)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main() 