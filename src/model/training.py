import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
import math
import argparse
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from evaluation import evaluate_model, save_metrics, print_metrics

class NGramModel:
    def __init__(self, n: int = 7, smoothing_k: float = 0.1):
        self.n = n
        self.smoothing_k = smoothing_k
        self.ngrams = defaultdict(lambda: defaultdict(float)) 
        self.vocab = set()
        self.context_counts = defaultdict(float)
        
    def tokenize_code(self, code: str) -> List[str]:
        # Tokenize Java code using Pygments.
        lexer = get_lexer_by_name('java')
        tokens = []
        
        for ttype, value in lexer.get_tokens(code):
            if ttype in Token.Comment or ttype in Token.Text.Whitespace:
                continue
            tokens.append(value)
            
        return tokens

    def pad_tokens(self, tokens: List[str]) -> List[str]:
        # Add start and end tokens to a sequence
        return ['<START>'] * (self.n - 1) + tokens + ['<END>']

    def train(self, methods: List[str]):
        print("\nProcessing training methods...")
        for method in tqdm(methods, desc="Training"):
            tokens = self.tokenize_code(method)
            padded_tokens = self.pad_tokens(tokens)
            
            for i in range(len(padded_tokens) - self.n + 1):
                context = tuple(padded_tokens[i:i + self.n - 1])
                target = padded_tokens[i + self.n - 1]
                
                self.ngrams[context][target] += 1
                self.context_counts[context] += 1
                self.vocab.add(target)

        print("\nApplying smoothing...")
        for context in tqdm(self.ngrams, desc="Smoothing"):
            for token in self.vocab:
                self.ngrams[context][token] += self.smoothing_k
                self.context_counts[context] += self.smoothing_k

    def get_probability(self, context: Tuple[str, ...], token: str) -> float:
        """Calculate smoothed probability P(token|context)."""
        if context not in self.context_counts:
            return 1.0 / len(self.vocab) if self.vocab else 0.0
            
        count_context = self.context_counts[context]
        count_ngram = self.ngrams[context][token]
        
        return count_ngram / count_context

    def predict_next(self, context: Tuple[str, ...]) -> Optional[Tuple[str, float]]:
        """Predict next token with smoothed probabilities."""
        if not self.vocab:
            return None
            
        predictions = {
            token: self.get_probability(context, token)
            for token in self.vocab
        }
        
        return max(predictions.items(), key=lambda x: x[1])

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate N-gram model')
    parser.add_argument('--data', type=str, default='./data/processed_methods.csv')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--n', type=int, default=7)
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--overwrite_metrics', type=bool, default=True,
                      help='Whether to overwrite existing metrics.json')
    parser.add_argument('--eval', action='store_true',
                      help='Run in evaluation mode with limited samples')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and split data
    print("Loading and preparing data...")
    methods = pd.read_csv(args.data)["Method Code No Comments"].dropna().tolist()
    
    # Limit samples in eval mode
    if args.eval and len(methods) > 500:
        print("\nRunning in evaluation mode - limiting to 500 samples")
        methods = methods[:500]
        
    train_methods, test_methods = train_test_split(methods, test_size=0.2, random_state=42)
    
    print(f"\nData Split:")
    print(f"Training set: {len(train_methods)} methods")
    print(f"Test set: {len(test_methods)} methods")
    
    # Train model
    model = NGramModel(n=args.n, smoothing_k=args.smoothing)
    print(f"\nTraining {args.n}-gram model...")
    model.train(train_methods)
    
    # Evaluate model
    metrics = evaluate_model(model, test_methods)
    print_metrics(metrics)
    save_metrics(metrics, args.output_dir, args.overwrite_metrics)
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main() 