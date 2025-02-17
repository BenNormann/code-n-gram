from data_processing.mining import extract_methods_to_csv
from data_processing.preprocessing import clean_and_tokenize_dataset
from model.training import NGramModel
import pandas as pd

def main():
    # Extract methods from repository
    repo_path = "path/to/your/python/repo"
    output_csv = "extracted_methods.csv"
    extract_methods_to_csv(repo_path, output_csv)
    
    # Load and preprocess the data
    df = pd.read_csv(output_csv)
    processed_df = clean_and_tokenize_dataset(df)
    
    # Train the model
    model = NGramModel(n=3)
    model.train(processed_df["Tokens"].tolist())
    
    # Generate some code
    generated_tokens = model.generate_sequence()
    print("Generated code tokens:", ' '.join(generated_tokens))

if __name__ == "__main__":
    main() 