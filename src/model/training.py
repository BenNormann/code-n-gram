"""
N-gram model for code completion.
Provides training and prediction functionality with progress tracking.
"""

import os
import pickle
from typing import List, Tuple, Dict, Optional, Set
from tqdm import tqdm

from model.data_handling import pad_tokens


class NGramModel:
    """N-gram language model for code completion."""
    
    def __init__(self, n: int = 4, smoothing_k: float = 0.01, lambda_factor: float = 0.4):
        """
        Initialize model with n-gram size and smoothing parameter.
        
        @input n: The n-gram size
        @input smoothing_k: The smoothing parameter
        @input lambda_factor: Interpolation factor for backoff (0-1)
        """
        if n < 1 or smoothing_k < 0:
            raise ValueError("n must be ≥1 and smoothing_k must be ≥0")
            
        self.n = n
        self.smoothing_k = smoothing_k
        self.lambda_factor = lambda_factor
        self.ngrams: Dict[Tuple[str, ...], Dict[str, int]] = {}
        self.vocab: Set[str] = set()
        self.context_counts: Dict[Tuple[str, ...], int] = {}
        
    def train(self, tokenized_methods: List[List[str]]):
        """
        Train model on a list of tokenized code methods.
        
        This method builds an n-gram language model for code completion.
        For an n-gram model (e.g., n=7), we count occurrences of all contexts
        of length 1 to n-1 and their following tokens to calculate conditional
        probabilities P(token | context).
        
        @input tokenized_methods: List of tokenized methods
        """
        if not tokenized_methods:
            raise ValueError("No valid methods to train on")
            
        # Build vocabulary from all tokens in the training data
        print("Building vocabulary...")
        for tokens in tokenized_methods:
            self.vocab.update(tokens)
        # Add special tokens for sequence boundaries
        self.vocab.update(['<START>', '<END>'])
        
        # Train the n-gram model with progress tracking
        print(f"\nTraining {self.n}-gram model...")
        for tokens in tqdm(tokenized_methods, desc=f"Training n={self.n}"):
            padded = pad_tokens(tokens, self.n)
            
            # For each position in the padded token sequence
            for i in range(len(padded) - self.n + 1):
                # Build all n-gram orders from 1 to n
                # This enables backoff to lower-order models during prediction
                for order in range(1, self.n + 1):
                    if i + order <= len(padded):
                        # Extract context (preceding tokens) and target token
                        context = tuple(padded[i:i + order - 1])  # Context of length (order-1)
                        target = padded[i + order - 1]            # Target token to predict
                        
                        # Initialize context entry if first occurrence
                        if context not in self.ngrams:
                            self.ngrams[context] = {}
                        
                        # Count target token occurrences for this context
                        self.ngrams[context][target] = self.ngrams[context].get(target, 0) + 1
                        
                        # Count total occurrences of this context
                        self.context_counts[context] = self.context_counts.get(context, 0) + 1

    def get_raw_probability(self, context: Tuple[str, ...], token: str) -> float:
        """
        Calculate raw probability without backoff.
        
        @input context: The context tuple
        @input token: The token to calculate probability for
        @return: The raw probability
        """
        if context not in self.context_counts:
            return 0.0
            
        count_context = self.context_counts[context]
        count_ngram = self.ngrams[context].get(token, 0)
        
        # Apply smoothing
        smoothed_count = count_ngram + self.smoothing_k
        smoothed_total = count_context + (self.smoothing_k * len(self.vocab))
        
        return smoothed_count / smoothed_total

    def get_probability(self, context: Tuple[str, ...], token: str) -> float:
        """
        Calculate smoothed probability P(token|context) with interpolation.
        
        @input context: The context tuple
        @input token: The token to calculate probability for
        @return: The probability of the token given the context
        """
        # Base case: unigram model or empty context
        if not context:
            # Uniform distribution if no context data
            if () not in self.context_counts:
                return 1.0 / len(self.vocab) if self.vocab else 0.0
                
            # Otherwise use unigram probability
            count_total = self.context_counts[()]
            count_token = self.ngrams[()].get(token, 0)
            return (count_token + self.smoothing_k) / (count_total + self.smoothing_k * len(self.vocab))
        
        # Get probability for current context
        higher_prob = self.get_raw_probability(context, token)
        
        # Interpolate with lower-order model
        lower_context = context[1:] if len(context) > 1 else ()
        lower_prob = self.get_probability(lower_context, token)
        
        # Interpolate between higher and lower order models
        # Use more weight on higher-order model when we have data
        if context in self.context_counts:
            # Adjust lambda based on context frequency
            context_weight = min(1.0, self.context_counts[context] / 10.0)
            effective_lambda = self.lambda_factor + (1 - self.lambda_factor) * context_weight
            return effective_lambda * higher_prob + (1 - effective_lambda) * lower_prob
        else:
            # Fall back to lower-order model if no data for this context
            return lower_prob

    def predict_next(self, context: Tuple[str, ...]) -> Optional[Tuple[str, float]]:
        """
        Predict most likely next token and its probability.
        
        @input context: The context tuple
        @return: Tuple of (predicted_token, probability) or None if no prediction
        """
        if not self.vocab:
            return None
            
        # Optimization: First check if we have this exact context
        if context in self.ngrams:
            # Get the most frequent token for this context
            best_token = max(self.ngrams[context].items(), key=lambda x: x[1])[0]
            prob = self.get_probability(context, best_token)
            return (best_token, prob)
            
        # If we don't have the exact context, check if we have a shorter context
        if len(context) > 1:
            shorter_context = context[1:]
            return self.predict_next(shorter_context)
            
        # If we have no context data at all, use the most frequent token overall
        if () in self.ngrams:
            best_token = max(self.ngrams[()].items(), key=lambda x: x[1])[0]
            prob = self.get_probability(context, best_token)
            return (best_token, prob)
            
        # Fallback to a random token from vocabulary
        best_token = next(iter(self.vocab))
        return (best_token, 1.0 / len(self.vocab))


def save_model(model: NGramModel, output_dir: str, filename: str = "best_model.pkl"):
    """
    Save model to disk using pickle.
    
    @input model: The model to save
    @input output_dir: Directory to save the model in
    @input filename: Filename for the saved model
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, filename)
    
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")


def load_model(model_path: str) -> Optional[NGramModel]:
    """
    Load model from disk.
    
    @input model_path: Path to the saved model
    @return: Loaded model or None if loading failed
    """
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
        
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None 