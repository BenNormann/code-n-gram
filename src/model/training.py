from collections import defaultdict
import numpy as np
from typing import List, Dict, Tuple

class NGramModel:
    def __init__(self, n: int = 3):
        self.n = n
        self.ngrams = defaultdict(lambda: defaultdict(int))
        self.vocab = set()

    def train(self, token_sequences: List[List[str]]):
        """Train the n-gram model on token sequences."""
        for sequence in token_sequences:
            # Add padding
            padded_seq = ['<START>'] * (self.n - 1) + sequence + ['<END>']
            
            # Build n-grams
            for i in range(len(padded_seq) - self.n + 1):
                context = tuple(padded_seq[i:i + self.n - 1])
                target = padded_seq[i + self.n - 1]
                self.ngrams[context][target] += 1
                self.vocab.add(target)

    def predict_next(self, context: Tuple[str, ...]) -> str:
        """Predict the next token given a context."""
        if context in self.ngrams:
            predictions = self.ngrams[context]
            return max(predictions.items(), key=lambda x: x[1])[0]
        return '<UNK>'

    def generate_sequence(self, max_length: int = 100) -> List[str]:
        """Generate a sequence of tokens."""
        context = tuple(['<START>'] * (self.n - 1))
        sequence = []
        
        for _ in range(max_length):
            next_token = self.predict_next(context)
            if next_token == '<END>':
                break
            sequence.append(next_token)
            context = tuple(list(context[1:]) + [next_token])
            
        return sequence 