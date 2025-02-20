import pandas as pd
import os
import re
from pygments.lexers import get_lexer_by_name
from pygments.token import Token

def remove_duplicates(data: pd.DataFrame, method_column: str = "Method Code") -> pd.DataFrame:
    """Remove duplicate methods based on method content.
    Almost Type-1 with the exception of comments.
    """
    return data.drop_duplicates(subset=method_column, keep="first")

def filter_ascii_methods(data: pd.DataFrame, method_column: str = "Method Code") -> pd.DataFrame:
    """Filter methods to include only those with ASCII characters."""
    data = data[data[method_column].apply(lambda x: all(ord(char) < 128 for char in x))]
    return data

def remove_outliers(data: pd.DataFrame, method_column: str = "Method Code", 
                   lower_percentile: int = 5, upper_percentile: int = 95) -> pd.DataFrame:
    """Remove outliers based on method length."""
    method_lengths = data[method_column].apply(len)
    lower_bound = method_lengths.quantile(lower_percentile / 100)
    upper_bound = method_lengths.quantile(upper_percentile / 100)
    return data[(method_lengths >= lower_bound) & (method_lengths <= upper_bound)]

def remove_boilerplate_methods(data: pd.DataFrame, method_column: str = "Method Code") -> pd.DataFrame:
    """Remove boilerplate methods like setters and getters."""
    boilerplate_patterns = [
        r"\bset[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",  # Setter methods
        r"\bget[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",  # Getter methods
    ]
    boilerplate_regex = re.compile("|".join(boilerplate_patterns))
    data = data[~data[method_column].apply(lambda x: bool(boilerplate_regex.search(x)))]
    return data

def remove_comments_from_dataframe(df: pd.DataFrame, method_column: str, language: str) -> pd.DataFrame:
    """
    Removes comments from code in a DataFrame and adds a new column with cleaned methods.

    Args:
        df (pd.DataFrame): DataFrame containing the methods
        method_column (str): Column name containing the raw code
        language (str): Programming language for the lexer (e.g., 'java')

    Returns:
        pd.DataFrame: Updated DataFrame with a new column '{method_column} No Comments'
    """
    def remove_comments(code: str) -> str:
        lexer = get_lexer_by_name(language)
        tokens = lexer.get_tokens(code)
        # Filter out comments
        clean_code = ''.join(token[1] for token in tokens 
                           if not (lambda t: t[0] in Token.Comment)(token))
        return clean_code

    # Create new column name
    clean_column = f"{method_column} No Comments"
    df[clean_column] = df[method_column].apply(remove_comments)
    return df

def clean_method_code(code: str) -> str:
    """Clean and standardize method code."""
    # Remove empty lines
    code = '\n'.join(line for line in code.splitlines() if line.strip())
    
    # Standardize whitespace while preserving necessary spacing
    code = re.sub(r'\s+', ' ', code)
    
    return code.strip()

def preprocess_methods(input_csv: str, output_csv: str, language: str = "java") -> None:
    """
    Preprocess methods from the input CSV and save to output CSV.
    Applies multiple cleaning and filtering steps.
    
    Args:
        input_csv (str): Path to input CSV with raw methods
        output_csv (str): Path to save processed methods
        language (str): Programming language of the methods (default: "java")
    """
    print(f"Reading methods from: {input_csv}")
    
    # Read the CSV file
    df = pd.read_csv(input_csv)
    print(f"Initial dataset size: {len(df)}")
    
    # Create a copy of the original method code
    method_column = "Method Code"
    df[f"{method_column} Original"] = df[method_column]
    
    # Remove comments
    df = remove_comments_from_dataframe(df, method_column, language)
    print(f"After removing comments: {len(df)}")
    
    # Clean and standardize code
    df[method_column] = df[f"{method_column} No Comments"].apply(clean_method_code)
    print(f"After cleaning code: {len(df)}")
    
    # Remove empty or invalid methods
    df = df[df[method_column].str.len() > 0]
    print(f"After removing empty methods: {len(df)}")
    
    # Apply various filtering steps
    df = remove_duplicates(df, method_column)
    print(f"After removing duplicates: {len(df)}")
    
    df = filter_ascii_methods(df, method_column)
    print(f"After filtering ASCII methods: {len(df)}")
    
    df = remove_outliers(df, method_column)
    print(f"After removing outliers: {len(df)}")
    
    df = remove_boilerplate_methods(df, method_column)
    print(f"After removing boilerplate methods: {len(df)}")
    
    # Save processed methods
    df.to_csv(output_csv, index=False)
    print(f"\nProcessed methods saved to: {output_csv}")
    print(f"Final dataset size: {len(df)}")

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
    
    preprocess_methods(input_csv, output_csv, language="java") 