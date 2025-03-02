"""
Data handling utilities for code completion model.
Includes tokenization, padding, and data splitting.
"""

import os
import pandas as pd
from typing import List, Tuple
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def ensure_directory_exists(directory: str) -> bool:
    """
    Ensure that the specified directory exists, creating it if necessary.
    
    @input directory: Path to the directory to check/create
    @return: True if the directory exists or was created, False otherwise
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory}: {str(e)}")
        return False


def tokenize_code(code: str, language: str = 'java') -> List[str]:
    """
    Convert code string into list of tokens, skipping comments and whitespace.
    
    @input code: The code string to tokenize
    @input language: The programming language of the code
    @return: List of tokens
    """
    if not code or not isinstance(code, str):
        return []
        
    try:
        return [
            value for ttype, value in get_lexer_by_name(language).get_tokens(code)
            if ttype not in Token.Comment and ttype not in Token.Text.Whitespace
        ]
    except Exception:
        return []


def pad_tokens(tokens: List[str], n: int) -> List[str]:
    """
    Add start and end tokens to a sequence.
    
    @input tokens: List of tokens to pad
    @input n: The n-gram size
    @return: Padded list of tokens
    """
    return ['<START>'] * (n - 1) + tokens + ['<END>']


def load_and_tokenize_data(data_path: str, progress_bar: bool = True) -> List[List[str]]:
    """
    Load methods from CSV file and tokenize them.
    
    @input data_path: Path to the CSV file containing method data
    @input progress_bar: Whether to show a progress bar
    @return: List of tokenized methods
    """
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        return []
        
    try:
        df = pd.read_csv(data_path)
        if "Method Code No Comments" not in df.columns:
            print(f"Error: Data file missing required column 'Method Code No Comments'")
            return []
    except Exception as e:
        print(f"Error reading data file: {str(e)}")
        return []
        
    methods = df["Method Code No Comments"].dropna().tolist()
    
    if not methods:
        print("No valid methods found in data file")
        return []
    
    all_tokenized = []
    iterator = tqdm(methods, desc="Tokenizing") if progress_bar else methods
    
    for method in iterator:
        tokens = tokenize_code(method)
        if tokens:
            all_tokenized.append(tokens)
    
    return all_tokenized


def split_data(tokenized_methods: List[List[str]], 
               test_size: float = 0.2,
               random_seed: int = 42) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    """
    Split data into training (80%), validation (10%), and test (10%) sets.
    
    @input tokenized_methods: List of tokenized methods
    @input test_size: Fraction of data to use for testing and validation
    @input random_seed: Random seed for reproducibility
    @return: Tuple of (train_data, val_data, test_data)
    """
    if not tokenized_methods:
        return [], [], []
    
    # First split off 80% for training
    train_data, remaining_data = train_test_split(tokenized_methods, test_size=test_size, random_state=random_seed)
    
    # Split remaining 20% into equal parts for validation and testing
    val_data, test_data = train_test_split(remaining_data, test_size=0.5, random_state=random_seed)
    
    return train_data, val_data, test_data 