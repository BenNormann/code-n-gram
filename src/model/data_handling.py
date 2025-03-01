"""
Data handling utilities for code completion model.
Includes tokenization, padding, validation checks, and data splitting.
"""

import os
import pandas as pd
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def validate_data_file(data_path: str) -> bool:
    """
    Validate that the data file exists and has the expected format.
    
    @input data_path: Path to the CSV file containing method data
    @return: True if the file is valid, False otherwise
    """
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        return False
        
    try:
        df = pd.read_csv(data_path)
        if "Method Code No Comments" not in df.columns:
            print(f"Error: Data file missing required column 'Method Code No Comments'")
            return False
            
        if df["Method Code No Comments"].dropna().empty:
            print(f"Error: Data file contains no valid methods")
            return False
            
        return True
    except Exception as e:
        print(f"Error validating data file: {str(e)}")
        return False


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
    if not validate_data_file(data_path):
        return []
        
    df = pd.read_csv(data_path)
    methods = df["Method Code No Comments"].dropna().tolist()
    
    if not methods:
        return []
    
    all_tokenized = []
    iterator = tqdm(methods, desc="Tokenizing") if progress_bar else methods
    
    for method in iterator:
        tokens = tokenize_code(method)
        if tokens:
            all_tokenized.append(tokens)
    
    return all_tokenized


def split_data(tokenized_methods: List[List[str]], 
               sample_size: Optional[int] = None, 
               random_seed: int = 42) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    """
    Split data into training (80%), validation (10%), and test (10%) sets.
    Optionally sample a subset of the data.
    
    @input tokenized_methods: List of tokenized methods
    @input sample_size: Number of methods to sample (None for all)
    @input random_seed: Random seed for reproducibility
    @return: Tuple of (train_data, val_data, test_data)
    """
    if not tokenized_methods:
        return [], [], []
    
    # Sample data if requested using Python's random module instead of numpy
    # This avoids the issue with numpy trying to create arrays of different shapes
    if sample_size and len(tokenized_methods) > sample_size:
        random.seed(random_seed)
        sampled_indices = random.sample(range(len(tokenized_methods)), sample_size)
        tokenized_methods = [tokenized_methods[i] for i in sampled_indices]
    
    # First split off 80% for training
    train_data, remaining_data = train_test_split(tokenized_methods, test_size=0.2, random_state=random_seed)
    
    # Split remaining 20% into equal parts for validation and testing
    val_data, test_data = train_test_split(remaining_data, test_size=0.5, random_state=random_seed)
    
    return train_data, val_data, test_data 