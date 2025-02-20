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

class NGramModel:
    def __init__(self, n: int = 7, smoothing_k: float = 0.1):
        self.n = n
        self.smoothing_k = smoothing_k  # Add-k smoothing parameter
        self.ngrams = defaultdict(lambda: defaultdict(float))  # Changed to float for smoothing
        self.vocab = set()
        self.context_counts = defaultdict(float)  # Changed to float for smoothing
        
    def tokenize_code(self, code: str) -> List[str]:
        """Tokenize Java code using Pygments."""
        lexer = get_lexer_by_name('java')
        tokens = []
        
        for ttype, value in lexer.get_tokens(code):
            if ttype in Token.Comment or ttype in Token.Text.Whitespace:
                continue
            tokens.append(value)
            
        return tokens

    def pad_tokens(self, tokens: List[str]) -> List[str]:
        """Add start and end tokens to a sequence."""
        return ['<START>'] * (self.n - 1) + tokens + ['<END>']

    def train(self, methods: List[str]):
        """Train the n-gram model with smoothing."""
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

    def evaluate(self, test_methods: List[str]) -> Dict:
        """Evaluate model with proper padding and perplexity calculation."""
        correct_predictions = 0
        total_predictions = 0
        total_log_prob = 0
        
        print("\nEvaluating on test methods...")
        for method in tqdm(test_methods, desc="Evaluating"):
            tokens = self.tokenize_code(method)
            padded_tokens = self.pad_tokens(tokens)
            
            for i in range(len(padded_tokens) - self.n + 1):
                context = tuple(padded_tokens[i:i + self.n - 1])
                actual = padded_tokens[i + self.n - 1]
                prediction = self.predict_next(context)
                
                if prediction:
                    pred_token, prob = prediction
                    if actual == pred_token:
                        correct_predictions += 1
                    total_predictions += 1
                    
                    if prob > 0:
                        total_log_prob += math.log2(prob)
                    else:
                        total_log_prob += math.log2(self.smoothing_k)

        # Calculate metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        perplexity = 2 ** (-total_log_prob / total_predictions) if total_predictions > 0 else float('inf')
        
        return {
            'accuracy': accuracy,
            'perplexity': perplexity,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'vocabulary_size': len(self.vocab)
        }

    def plot_results(self, metrics: Dict, output_dir: str):
        """Plot evaluation results."""
        plt.figure(figsize=(8, 6))
        metrics_to_plot = {
            'Accuracy': metrics['accuracy'],
            'Normalized Perplexity': min(1.0, 1/metrics['perplexity'])  # Normalize for visualization
        }
        
        sns.barplot(x=list(metrics_to_plot.keys()), y=list(metrics_to_plot.values()))
        plt.title('Model Performance Metrics')
        plt.ylim(0, 1)
        for i, v in enumerate(metrics_to_plot.values()):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        plt.savefig(os.path.join(output_dir, 'performance_metrics.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate N-gram model')
    parser.add_argument('--data', type=str, default='./data/processed_methods.csv')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--n', type=int, default=7)
    parser.add_argument('--smoothing', type=float, default=0.1)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and split data
    print("Loading and preparing data...")
    methods = pd.read_csv(args.data)["Method Code No Comments"].dropna().tolist()
    train_methods, test_methods = train_test_split(methods, test_size=0.2, random_state=42)
    
    print(f"\nData Split:")
    print(f"Training set: {len(train_methods)} methods")
    print(f"Test set: {len(test_methods)} methods")
    
    # Train and evaluate
    model = NGramModel(n=args.n, smoothing_k=args.smoothing)
    print(f"\nTraining {args.n}-gram model...")
    model.train(train_methods)
    
    print("\nEvaluating model...")
    metrics = model.evaluate(test_methods)
    
    # Print results
    print("\nResults:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Perplexity: {metrics['perplexity']:.3f}")
    print(f"Vocabulary size: {metrics['vocabulary_size']}")
    print(f"Total predictions: {metrics['total_predictions']}")
    print(f"Correct predictions: {metrics['correct_predictions']}")
    
    # Save results
    model.plot_results(metrics, args.output_dir)
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main() 