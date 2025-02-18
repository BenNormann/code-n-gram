import pandas as pd
import os
from typing import List, Dict
import re

def clean_method_code(code: str) -> str:
    """Clean and standardize Java method code."""
    # Remove comments
    code = re.sub(r'//.*?\n|/\*.*?\*/', '', code, flags=re.DOTALL)
    
    # Remove empty lines
    code = '\n'.join(line for line in code.splitlines() if line.strip())
    
    # Standardize whitespace
    code = re.sub(r'\s+', ' ', code)
    
    return code.strip()

def preprocess_methods(input_csv: str, output_csv: str) -> None:
    """
    Preprocess Java methods from the input CSV and save to output CSV.
    
    Args:
        input_csv (str): Path to input CSV with raw methods
        output_csv (str): Path to save processed methods
    """
    print(f"Reading methods from: {input_csv}")
    
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Clean method code
    df['Method Code'] = df['Method Code'].apply(clean_method_code)
    
    # Remove empty or invalid methods
    df = df[df['Method Code'].str.len() > 0]
    
    # Remove duplicate methods
    df = df.drop_duplicates(subset=['Method Code'])
    
    # Save processed methods
    df.to_csv(output_csv, index=False)
    print(f"Processed methods saved to: {output_csv}")
    print(f"Total methods after preprocessing: {len(df)}")

if __name__ == "__main__":
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Set up input/output paths
    data_dir = os.path.join(project_root, "data")
    input_csv = os.path.join(data_dir, "extracted_methods.csv")
    output_csv = os.path.join(data_dir, "processed_methods.csv")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.exists(input_csv):
        print(f"Error: Input file not found: {input_csv}")
        exit(1)
    
    preprocess_methods(input_csv, output_csv) 