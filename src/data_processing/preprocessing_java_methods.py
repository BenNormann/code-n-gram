"""
Preprocessing script for Java methods that produces high-quality
method code with comments removed.
"""

import os
import sys
import argparse
import pandas as pd
import re
from tqdm import tqdm
from pygments.lexers import get_lexer_by_name
from pygments.token import Token

def ensure_directory_exists(directory: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def remove_duplicates(data: pd.DataFrame, method_column: str) -> pd.DataFrame:
    """Remove duplicate methods based on method content."""
    initial_size = len(data)
    data = data.drop_duplicates(subset=method_column, keep="first")
    print(f"Removed {initial_size - len(data)} duplicate methods")
    return data

def filter_ascii_methods(data: pd.DataFrame, method_column: str) -> pd.DataFrame:
    """Filter methods to include only those with ASCII characters."""
    initial_size = len(data)
    data = data[data[method_column].apply(lambda x: all(ord(char) < 128 for char in str(x)))]
    print(f"Removed {initial_size - len(data)} non-ASCII methods")
    return data

def remove_outliers(data: pd.DataFrame, method_column: str, 
                   lower_percentile: int = 5, upper_percentile: int = 95) -> pd.DataFrame:
    """Remove outliers based on method length using a distribution-based approach."""
    initial_size = len(data)
    method_lengths = data[method_column].apply(len)
    lower_bound = method_lengths.quantile(lower_percentile / 100)
    upper_bound = method_lengths.quantile(upper_percentile / 100)
    
    data = data[(method_lengths >= lower_bound) & (method_lengths <= upper_bound)]
    print(f"Removed {initial_size - len(data)} outlier methods")
    print(f"Length bounds: {lower_bound:.0f} to {upper_bound:.0f} characters")
    return data

def remove_boilerplate_methods(data: pd.DataFrame, method_column: str) -> pd.DataFrame:
    """Remove boilerplate methods like setters, getters, and other common patterns."""
    initial_size = len(data)
    
    boilerplate_patterns = [
        r"\bset[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",  # Setter methods
        r"\bget[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",  # Getter methods
        r"\bto(String|Array)\(\)\s*{",          # toString/toArray methods
        r"\bhash(Code|)\(\)\s*{",               # hashCode methods
        r"\bequals\(Object\s+\w+\)\s*{",        # equals methods
        r"\bclone\(\)\s*{",                     # clone methods
        r"^\s*@Override\s*\n",                  # Overridden methods
    ]
    boilerplate_regex = re.compile("|".join(boilerplate_patterns))
    data = data[~data[method_column].apply(lambda x: bool(boilerplate_regex.search(str(x))))]
    print(f"Removed {initial_size - len(data)} boilerplate methods")
    return data

def remove_comments_from_code(code: str, language: str = "java") -> str:
    """
    Removes comments from code and formats with proper indentation.
    Uses Pygments for accurate tokenization and comment removal.
    
    Args:
        code: The source code string
        language: Programming language of the code
        
    Returns:
        Formatted code with comments removed
    """
    if not isinstance(code, str) or not code.strip():
        return ''
        
    lexer = get_lexer_by_name(language)
    tokens = []
    current_line = []
    formatted_lines = []
    indent_level = 0
    
    for ttype, value in lexer.get_tokens(code):
        # Skip comments
        if Token.Comment in ttype or 'Comment' in str(ttype):
            continue
            
        # Handle different token types
        if Token.Text in ttype:
            if '\n' in value:
                # Process the current line
                if current_line:
                    line = ''.join(current_line).strip()
                    if line:
                        formatted_lines.append('    ' * indent_level + line)
                current_line = []
            elif not value.isspace():
                current_line.append(value)
        else:
            # Handle braces for indentation
            if value == '{':
                current_line.append(' {')
                # Process the current line
                line = ''.join(current_line).strip()
                if line:
                    formatted_lines.append('    ' * indent_level + line)
                current_line = []
                indent_level += 1
            elif value == '}':
                # Process any content before the closing brace
                if current_line:
                    line = ''.join(current_line).strip()
                    if line:
                        formatted_lines.append('    ' * indent_level + line)
                current_line = []
                indent_level = max(0, indent_level - 1)
                formatted_lines.append('    ' * indent_level + '}')
            elif value == ';':
                current_line.append(';')
                # Process the current line
                line = ''.join(current_line).strip()
                if line:
                    formatted_lines.append('    ' * indent_level + line)
                current_line = []
            else:
                # Add space before and after operators
                if ttype in Token.Operator:
                    current_line.append(f' {value} ')
                else:
                    current_line.append(value)
    
    # Process any remaining content
    if current_line:
        line = ''.join(current_line).strip()
        if line:
            formatted_lines.append('    ' * indent_level + line)
    
    # Join lines with proper line breaks
    return '\n'.join(formatted_lines)

def clean_method_code(code: str) -> str:
    """Clean and standardize method code."""
    if not isinstance(code, str) or not code.strip():
        return ''
        
    # Remove empty lines and normalize whitespace
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    lines = [' '.join(line.split()) for line in lines]
    
    # Join with appropriate spacing
    return '\n'.join(lines)

def process_methods(input_csv: str, output_csv: str, language: str = "java") -> None:
    """
    Process methods from the input CSV and save to output CSV with only
    the "Method Code No Comments" column.
    
    Args:
        input_csv: Path to input CSV with raw methods
        output_csv: Path to save processed methods
        language: Programming language of the methods
    """
    print(f"\nReading methods from: {input_csv}")
    
    try:
        df = pd.read_csv(input_csv)
        print(f"Initial dataset size: {len(df)}")
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        sys.exit(1)
    
    # Determine the method code column name from the input dataframe
    if "Method Code" in df.columns:
        method_column = "Method Code"
    elif "method_code" in df.columns:
        method_column = "method_code"
    else:
        # Attempt to find a column with "code" in the name
        code_columns = [col for col in df.columns if "code" in col.lower()]
        if code_columns:
            method_column = code_columns[0]
        else:
            print("Error: Could not identify method code column in input CSV")
            sys.exit(1)
    
    print(f"Using '{method_column}' as input column")
    
    # Process methods with progress reporting
    print("Removing comments from methods...")
    tqdm.pandas(desc="Processing")
    df["Method Code No Comments"] = df[method_column].progress_apply(
        lambda x: remove_comments_from_code(x, language)
    )
    
    # Remove methods that became empty after comment removal
    initial_size = len(df)
    df = df[df["Method Code No Comments"].str.len() > 0]
    print(f"Removed {initial_size - len(df)} methods that were only comments")
    
    # Clean the code
    print("Cleaning code...")
    df["Method Code No Comments"] = df["Method Code No Comments"].apply(clean_method_code)
    
    # Apply all quality filters
    df = remove_duplicates(df, "Method Code No Comments")
    df = filter_ascii_methods(df, "Method Code No Comments")
    df = remove_outliers(df, "Method Code No Comments")
    df = remove_boilerplate_methods(df, "Method Code No Comments")
    
    # Final check for empty methods
    df = df[df["Method Code No Comments"].str.len() > 0]
    
    # Only keep the "Method Code No Comments" column
    final_df = df[["Method Code No Comments"]]
    
    # Save processed methods
    final_df.to_csv(output_csv, index=False)
    print(f"\nProcessed methods saved to: {output_csv}")
    print(f"Final dataset size: {len(final_df)}")
    
    # Print a sample to verify content
    if len(final_df) > 0:
        print("\nSample processed method:")
        print(final_df["Method Code No Comments"].iloc[0])

def main():
    """Main function to handle command line arguments and execute preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess Java methods to produce high-quality code with comments removed."
    )
    parser.add_argument(
        "--input", type=str, default="./data/extracted_methods.csv",
        help="Path to input CSV file with method code"
    )
    parser.add_argument(
        "--output", type=str, default="./data/processed_methods.csv",
        help="Path to output CSV file with processed methods"
    )
    parser.add_argument(
        "--language", type=str, default="java",
        help="Programming language of the methods"
    )
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    ensure_directory_exists(output_dir)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Process the methods
    process_methods(args.input, args.output, args.language)

if __name__ == "__main__":
    main()