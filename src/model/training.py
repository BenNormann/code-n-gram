import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Optional
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
import argparse
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))

from evaluation import evaluate_model, save_metrics, print_metrics

class NGramModel:
    def __init__(self, n: int = 7, smoothing_k: float = 0.1, vocabulary: set = None):
        self.n = n
        self.smoothing_k = smoothing_k
        self.ngrams = defaultdict(lambda: defaultdict(float)) 
        self.vocab = vocabulary if vocabulary is not None else set()
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
        # If no vocabulary was provided, build it from the training data
        if not self.vocab:
            print("Building vocabulary...")
            for method in tqdm(methods, desc="Building Vocabulary"):
                tokens = self.tokenize_code(method)
                self.vocab.update(tokens)
            self.vocab.add('<START>')
            self.vocab.add('<END>')
            
        for method in tqdm(methods, desc="Training"):
            tokens = self.tokenize_code(method)
            padded_tokens = self.pad_tokens(tokens)
            
            for i in range(len(padded_tokens) - self.n + 1):
                context = tuple(padded_tokens[i:i + self.n - 1])
                target = padded_tokens[i + self.n - 1]
                
                self.ngrams[context][target] += 1
                self.context_counts[context] += 1

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

def build_vocabulary(methods: List[str]) -> set:
    """Build vocabulary from all methods."""
    print("\nBuilding global vocabulary...")
    vocab = set()
    
    for method in tqdm(methods, desc="Building Vocabulary"):
        lexer = get_lexer_by_name('java')
        for ttype, value in lexer.get_tokens(method):
            if ttype in Token.Comment or ttype in Token.Text.Whitespace:
                continue
            vocab.add(value)
    
    # Add special tokens
    vocab.add('<START>')
    vocab.add('<END>')
    return vocab

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
    
    # Load data
    print("Loading and preparing data...")
    methods = pd.read_csv(args.data)["Method Code No Comments"].dropna().tolist()
    
    # Limit samples in eval mode with random sampling
    if args.eval and len(methods) > 500:
        print("\nRunning in evaluation mode - randomly sampling 500 methods")
        methods = np.random.choice(methods, size=500, replace=False).tolist()

    # Split data into training, test, and evaluation sets
    print(f"\nData Split:")
    
    # First split off training set (80%)
    train_methods, remaining_methods = train_test_split(methods, test_size=0.2, random_state=42)
    # Split remaining 20% equally between test and evaluation
    test_methods, evaluation_methods = train_test_split(remaining_methods, test_size=0.5, random_state=42)
    
    print(f"Training set: {len(train_methods)} methods")
    print(f"Test set: {len(test_methods)} methods")
    print(f"Evaluation set: {len(evaluation_methods)} methods")
    
    # Build vocabulary from training data
    vocabulary = build_vocabulary(train_methods)
    print(f"\nVocabulary size: {len(vocabulary)}")
    
    # Train model with pre-built vocabulary
    model = NGramModel(n=args.n, smoothing_k=args.smoothing, vocabulary=vocabulary)
    print(f"\nTraining {args.n}-gram model...")
    model.train(train_methods)
    
    # Evaluate on evaluation set
    print("\nEvaluating on evaluation set:")
    metrics = evaluate_model(model, evaluation_methods)
    print_metrics(metrics)
    
    # Save metrics
    save_metrics(metrics, args.output_dir, args.overwrite_metrics)
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main() 