"""
Sequence Reversal Task: Reverse a sequence of tokens.

Proper next-token prediction format:
- Input:  [sequence] SEP [reversed_0, ..., reversed_{n-1}]
- Target: shifted by 1 for next-token prediction
"""

import random
from typing import Tuple, List
from .base import BaseTask


class ReversalTask(BaseTask):
    """
    Task: Reverse a sequence of tokens.
    
    Format: <input_sequence> SEP <reversed_sequence> EOS
    Uses proper next-token prediction (target shifted by 1).
    """
    
    def __init__(self, config):
        super().__init__(config)
        # Vocab: 0=PAD, 1=SEP, 2=EOS, 3+ = symbols
        self.EOS_TOKEN = 2
        self.NUM_OFFSET = 3
        self.num_symbols = config.vocab_size - self.NUM_OFFSET
        
    def get_vocab_size(self) -> int:
        return self.config.vocab_size
    
    def generate_sample(self, seq_length: int) -> Tuple[List[int], List[int]]:
        """Generate a reversal sample with proper next-token prediction."""
        # Generate random sequence
        sequence = [random.randint(0, self.num_symbols - 1) for _ in range(seq_length)]
        reversed_seq = list(reversed(sequence))
        
        # Convert to tokens
        input_tokens = [s + self.NUM_OFFSET for s in sequence]
        output_tokens = [s + self.NUM_OFFSET for s in reversed_seq]
        
        # Full sequence: input SEP output EOS
        full_seq = input_tokens + [self.SEP_TOKEN] + output_tokens + [self.EOS_TOKEN]
        
        # Input is everything except last token
        input_seq = full_seq[:-1]
        
        # Target is shifted: target[i] = full_seq[i+1]
        # Only predict after SEP
        target_seq = []
        sep_seen = False
        for i in range(len(input_seq)):
            if input_seq[i] == self.SEP_TOKEN:
                sep_seen = True
            if sep_seen:
                target_seq.append(full_seq[i + 1])
            else:
                target_seq.append(-100)
        
        return input_seq, target_seq
    
    def decode_sample(self, tokens: List[int]) -> str:
        """Decode tokens to readable format."""
        parts = []
        for t in tokens:
            if t == self.PAD_TOKEN:
                continue
            elif t == self.SEP_TOKEN:
                parts.append("|")
            elif t == self.EOS_TOKEN:
                parts.append("<EOS>")
            else:
                parts.append(str(t - self.NUM_OFFSET))
        return " ".join(parts)
