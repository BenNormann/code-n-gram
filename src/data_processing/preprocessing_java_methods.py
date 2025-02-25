import pandas as pd
import os
import re
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
from pygments.lexers.jvm import JavaLexer

def remove_duplicates(data: pd.DataFrame, method_column: str = "Method Code") -> pd.DataFrame:
    """Remove duplicate methods based on method content.
    Almost Type-1 with the exception of comments.
    """
    print(f"Removing duplicates...")
    initial_size = len(data)
    data = data.drop_duplicates(subset=method_column, keep="first")
    print(f"Removed {initial_size - len(data)} duplicate methods")
    return data

def filter_ascii_methods(data: pd.DataFrame, method_column: str = "Method Code") -> pd.DataFrame:
    """Filter methods to include only those with ASCII characters."""
    print(f"Filtering non-ASCII methods...")
    initial_size = len(data)
    data = data[data[method_column].apply(lambda x: all(ord(char) < 128 for char in str(x)))]
    print(f"Removed {initial_size - len(data)} non-ASCII methods")
    return data

def remove_outliers(data: pd.DataFrame, method_column: str = "Method Code", 
                   lower_percentile: int = 5, upper_percentile: int = 95) -> pd.DataFrame:
    """Remove outliers based on method length using a distribution-based approach."""
    print(f"Removing length outliers...")
    initial_size = len(data)
    method_lengths = data[method_column].apply(len)
    lower_bound = method_lengths.quantile(lower_percentile / 100)
    upper_bound = method_lengths.quantile(upper_percentile / 100)
    
    # Filter methods based on length bounds
    data = data[(method_lengths >= lower_bound) & (method_lengths <= upper_bound)]
    print(f"Removed {initial_size - len(data)} outlier methods")
    print(f"Length bounds: {lower_bound:.0f} to {upper_bound:.0f} characters")
    return data

def remove_boilerplate_methods(data: pd.DataFrame, method_column: str = "Method Code") -> pd.DataFrame:
    """Remove boilerplate methods like setters, getters, and other common patterns."""
    print(f"Removing boilerplate methods...")
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

def remove_comments_from_dataframe(df: pd.DataFrame, method_column: str, language: str) -> pd.DataFrame:
    """
    Removes comments from code in a DataFrame and formats the methods with proper line breaks and indentation.
    Uses Pygments for accurate tokenization and comment removal.
    """
    print(f"Removing comments from methods...")
    initial_size = len(df)
    
    def remove_comments_and_format(code: str) -> str:
        if not isinstance(code, str):
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

    # Apply comment removal and formatting
    df = df.copy()
    df[method_column] = df[method_column].apply(remove_comments_and_format)
    
    # Remove methods that became empty after comment removal
    df = df[df[method_column].str.len() > 0]
    print(f"Removed {initial_size - len(df)} methods that were only comments")
    return df

def clean_method_code(code: str) -> str:
    """Clean and standardize method code."""
    if not isinstance(code, str):
        return ''
        
    # Remove empty lines and normalize whitespace
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    lines = [' '.join(line.split()) for line in lines]
    
    # Join with appropriate spacing
    return '\n'.join(lines)

def preprocess_methods(input_csv: str, output_csv: str, language: str = "java") -> None:
    """
    Preprocess methods from the input CSV and save to output CSV.
    Applies multiple cleaning and filtering steps.
    
    Args:
        input_csv (str): Path to input CSV with raw methods
        output_csv (str): Path to save processed methods
        language (str): Programming language of the methods (default: "java")
    """
    print(f"\nReading methods from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Initial dataset size: {len(df)}")
    
    # Define column names
    method_column = "Method Code"
    
    # First, remove comments directly from the Method Code column
    df = remove_comments_from_dataframe(df, method_column, language)
    print(f"After removing comments: {len(df)}")
    
    # Clean the code
    df[method_column] = df[method_column].apply(clean_method_code)
    print(f"After cleaning code: {len(df)}")
    
    # Apply filtering steps in sequence
    df = remove_duplicates(df, method_column)
    print(f"After removing duplicates: {len(df)}")
    
    df = filter_ascii_methods(df, method_column)
    print(f"After filtering ASCII methods: {len(df)}")
    
    df = remove_outliers(df, method_column)
    print(f"After removing outliers: {len(df)}")
    
    df = remove_boilerplate_methods(df, method_column)
    print(f"After removing boilerplate methods: {len(df)}")
    
    # Only keep necessary columns and ensure Method Code contains cleaned code
    final_columns = ["Branch Name", "Commit Hash", "File Name", "Method Name", 
                    method_column, "Commit Link"]
    
    # Only keep columns that exist in the DataFrame
    final_columns = [col for col in final_columns if col in df.columns]
    
    # Rename the Method Code column to match what training.py expects
    df = df.rename(columns={method_column: "Method Code No Comments"})
    
    # Update final_columns list with new column name
    final_columns = [col if col != method_column else "Method Code No Comments" for col in final_columns]
    
    # Save processed methods with only the necessary columns
    df[final_columns].to_csv(output_csv, index=False)
    print(f"\nProcessed methods saved to: {output_csv}")
    print(f"Final dataset size: {len(df)}")
    
    # Print a sample to verify content
    if len(df) > 0:
        print("\nSample processed method:")
        print(df["Method Code No Comments"].iloc[0])

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, "data")
    input_csv = os.path.join(data_dir, "extracted_methods.csv")
    output_csv = os.path.join(data_dir, "processed_methods.csv")
    os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.exists(input_csv):
        print(f"Error: Input file not found: {input_csv}")
        exit(1)
    
    preprocess_methods(input_csv, output_csv, language="java")