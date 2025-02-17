import pandas as pd
import re
import os

def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate methods based on method content.
    Almost Type-1 with the exception of comments
    """
    return data.drop_duplicates(subset="Method Code", keep="first")

def filter_ascii_methods(data: pd.DataFrame) -> pd.DataFrame:
    """Filter methods to include only those with ASCII characters."""
    return data[data["Method Code"].apply(lambda x: all(ord(char) < 128 for char in x))]

def remove_outliers(data: pd.DataFrame, lower_percentile: int = 5, 
                   upper_percentile: int = 95) -> pd.DataFrame:
    """Remove outliers based on method length."""
    method_lengths = data["Method Code"].apply(len)
    lower_bound = method_lengths.quantile(lower_percentile / 100)
    upper_bound = method_lengths.quantile(upper_percentile / 100)
    return data[(method_lengths >= lower_bound) & (method_lengths <= upper_bound)]

def remove_boilerplate_methods(data: pd.DataFrame) -> pd.DataFrame:
    """Remove common boilerplate Python methods."""
    def is_boilerplate(code: str) -> bool:
        patterns = [
            r"^def __init__\s*\([^)]*\)\s*:\s*pass\s*$",
            r"^def __str__\s*\([^)]*\)\s*:\s*return\s+['\"][^'\"]*['\"]$",
            r"^def get_[a-zA-Z_]+\s*\([^)]*\)\s*:\s*return\s+self\.[a-zA-Z_]+$",
            r"^def set_[a-zA-Z_]+\s*\([^)]*\)\s*:\s*self\.[a-zA-Z_]+\s*=\s*[a-zA-Z_]+$"
        ]
        return any(re.match(pattern, code.strip()) for pattern in patterns)
    
    return data[~data["Method Code"].apply(is_boilerplate)]

def remove_comments(code: str) -> str:
    """Remove comments from Python code."""
    # Remove multi-line comments
    code = re.sub(r'"""[\s\S]*?"""', '', code)
    code = re.sub(r"'''[\s\S]*?'''", '', code)
    
    # Remove single-line comments
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    
    # Remove extra whitespace
    code = '\n'.join(line.strip() for line in code.split('\n'))
    return code

if __name__ == "__main__":
    # Simplified path handling - read directly from data folder
    input_csv = "./data/extracted_methods.csv"
    output_csv = "./data/processed_methods.csv"

    # Check if input file exists
    if not os.path.exists(input_csv):
        print(f"Error: Input file not found at {input_csv}")
        print("Please run the mining step first using mining.py")
        exit(1)

    # Load the extracted methods
    print(f"Loading data from: {input_csv}")
    data = pd.read_csv(input_csv)
    print("Initial dataset size:", len(data))

    # Apply preprocessing steps
    print("Removing duplicates...")
    data = remove_duplicates(data)
    print("After removing duplicates:", len(data))

    print("Filtering ASCII methods...")
    data = filter_ascii_methods(data)
    print("After filtering ASCII methods:", len(data))

    print("Removing outliers...")
    data = remove_outliers(data)
    print("After removing outliers:", len(data))

    print("Removing boilerplate methods...")
    data = remove_boilerplate_methods(data)
    print("After removing boilerplate methods:", len(data))

    print("Cleaning comments...")
    data["Clean Code"] = data["Method Code"].apply(remove_comments)

    # Save processed data
    data.to_csv(output_csv, index=False)
    print(f"Processed data saved to: {output_csv}") 