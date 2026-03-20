"""
Modular Arithmetic Task: Compute (a + b + c + ...) mod p

Proper next-token prediction:
- Input:  n1 + n2 + ... + nk SEP
- Target: predict result token after SEP
"""

import random
from typing import Tuple, List
from .base import BaseTask


class ModularArithTask(BaseTask):
    """
    Task: Compute modular sum of a sequence of numbers.
    
    Format: n1 + n2 + ... + nk SEP result EOS
    Uses proper next-token prediction (target shifted by 1).
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.modulo = config.modulo  # e.g., 97
        
        # Vocab: 0=PAD, 1=SEP, 2=EOS, 3=PLUS, 4...(4+modulo-1) = numbers 0 to modulo-1
        self.EOS_TOKEN = 2
        self.PLUS_TOKEN = 3
        self.NUM_OFFSET = 4
        
    def get_vocab_size(self) -> int:
        # PAD, SEP, EOS, PLUS, and numbers 0 to modulo-1
        return self.NUM_OFFSET + self.modulo
    
    def generate_sample(self, seq_length: int) -> Tuple[List[int], List[int]]:
        """Generate a modular arithmetic sample with proper next-token prediction."""
        # Number of operands
        num_operands = seq_length
        
        # Generate random numbers
        numbers = [random.randint(0, self.modulo - 1) for _ in range(num_operands)]
        result = sum(numbers) % self.modulo
        
        # Build full sequence: n1 + n2 + n3 + ... SEP result EOS
        tokens = []
        for i, n in enumerate(numbers):
            tokens.append(n + self.NUM_OFFSET)
            if i < len(numbers) - 1:
                tokens.append(self.PLUS_TOKEN)
        
        tokens.append(self.SEP_TOKEN)
        tokens.append(result + self.NUM_OFFSET)
        tokens.append(self.EOS_TOKEN)
        
        # Input is everything except last token
        input_seq = tokens[:-1]
        
        # Target: only predict after SEP (the result and EOS)
        target_seq = []
        sep_seen = False
        for i in range(len(input_seq)):
            if input_seq[i] == self.SEP_TOKEN:
                sep_seen = True
            if sep_seen:
                target_seq.append(tokens[i + 1])
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
                parts.append("=")
            elif t == self.EOS_TOKEN:
                parts.append("<EOS>")
            elif t == self.PLUS_TOKEN:
                parts.append("+")
            else:
                parts.append(str(t - self.NUM_OFFSET))
        return " ".join(parts)
