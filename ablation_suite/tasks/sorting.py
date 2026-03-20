"""
Sorting Task: Sort a sequence of numbers.

Proper next-token prediction format:
- Input:  [unsorted] SEP [sorted_0, sorted_1, ..., sorted_{n-1}]
- Target: shifted by 1, so target[i] = input[i+1]
  Position before SEP predicts SEP
  Position at SEP predicts sorted_0
  Position at sorted_i predicts sorted_{i+1}
"""

import random
from typing import Tuple, List
from .base import BaseTask


class SortingTask(BaseTask):
    """
    Task: Sort a sequence of numbers in ascending order.
    
    Format: <input_numbers> SEP <sorted_numbers> EOS
    Uses proper next-token prediction (target shifted by 1).
    """
    
    def __init__(self, config):
        super().__init__(config)
        # Vocab: 0=PAD, 1=SEP, 2=EOS, 3+ = numbers
        self.EOS_TOKEN = 2
        self.NUM_OFFSET = 3
        self.max_num = config.vocab_size - self.NUM_OFFSET
        
    def get_vocab_size(self) -> int:
        return self.config.vocab_size
    
    def generate_sample(self, seq_length: int) -> Tuple[List[int], List[int]]:
        """Generate a sorting sample with proper next-token prediction."""
        # Generate random numbers (as token IDs)
        numbers = [random.randint(0, self.max_num - 1) for _ in range(seq_length)]
        sorted_numbers = sorted(numbers)
        
        # Convert to tokens
        input_tokens = [n + self.NUM_OFFSET for n in numbers]
        output_tokens = [n + self.NUM_OFFSET for n in sorted_numbers]
        
        # Full sequence: input SEP output EOS
        full_seq = input_tokens + [self.SEP_TOKEN] + output_tokens + [self.EOS_TOKEN]
        
        # Input is everything except last token
        input_seq = full_seq[:-1]
        
        # Target is shifted: target[i] = full_seq[i+1]
        # We only care about predicting after seeing SEP (the output portion)
        target_seq = []
        sep_seen = False
        for i in range(len(input_seq)):
            if input_seq[i] == self.SEP_TOKEN:
                sep_seen = True
            if sep_seen:
                # After SEP, predict the next token
                target_seq.append(full_seq[i + 1])
            else:
                # Before and at input portion, ignore
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
