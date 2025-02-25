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
    Removes comments from code in a DataFrame and adds a new column with cleaned methods.
    Uses Pygments for accurate tokenization and comment removal.
    """
    print(f"Removing comments from methods...")
    initial_size = len(df)
    
    def remove_comments(code: str) -> str:
        if not isinstance(code, str):
            return ''
            
        lexer = get_lexer_by_name(language)
        tokens = []
        last_token_type = None
        
        # First pass: collect tokens without comments
        for ttype, value in lexer.get_tokens(code):
            # Skip all types of comments
            if Token.Comment in ttype or 'Comment' in str(ttype):
                continue
                
            # Handle whitespace
            if ttype in Token.Text:
                if last_token_type and last_token_type not in Token.Text:
                    tokens.append(' ')
            else:
                tokens.append(value)
            
            last_token_type = ttype
        
        # Join tokens and do initial cleanup
        code_str = ''.join(tokens)
        
        # Format the code
        formatted_lines = []
        current_line = []
        depth = 0
        
        for char in code_str:
            if char == '{':
                current_line.append(' {')
                formatted_lines.append(''.join(current_line))
                current_line = []
                depth += 1
            elif char == '}':
                if current_line:
                    formatted_lines.append('    ' * depth + ''.join(current_line))
                depth = max(0, depth - 1)
                formatted_lines.append('    ' * depth + '}')
                current_line = []
            elif char == ';':
                current_line.append(';')
                formatted_lines.append('    ' * depth + ''.join(current_line))
                current_line = []
            elif char == '\n':
                if current_line:
                    formatted_lines.append('    ' * depth + ''.join(current_line))
                    current_line = []
            else:
                current_line.append(char)
        
        if current_line:
            formatted_lines.append('    ' * depth + ''.join(current_line))
        
        # Clean up extra whitespace while preserving structure
        formatted_lines = [line.strip() for line in formatted_lines]
        formatted_lines = [line for line in formatted_lines if line]
        
        # Add proper spacing around parentheses and operators
        result = []
        for line in formatted_lines:
            # Fix spacing around operators
            line = re.sub(r'([=<>!&|+\-*/%])', r' \1 ', line)
            line = re.sub(r'\s+([,;])', r'\1', line)
            line = re.sub(r'\(\s+', '(', line)
            line = re.sub(r'\s+\)', ')', line)
            line = re.sub(r'\s+', ' ', line)
            result.append(line.strip())
        
        return '\n'.join(result)

    # Apply comment removal
    df = df.copy()
    df[method_column] = df[method_column].apply(remove_comments)
    
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
    
    # Save processed methods with only the necessary columns
    df[final_columns].to_csv(output_csv, index=False)
    print(f"\nProcessed methods saved to: {output_csv}")
    print(f"Final dataset size: {len(df)}")
    
    # Print a sample to verify content
    if len(df) > 0:
        print("\nSample processed method:")
        print(df[method_column].iloc[0])

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