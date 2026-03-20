"""
Bitwise Addition Task: Add two binary numbers.

Input: a[n-1]...a[0] SEP b[n-1]...b[0] SEP
Output: c[n]...c[0] (n+1 bits for potential carry)

Tests algorithmic carry propagation.
OOD: Train on 16-bit, test on 32-bit and 48-bit.
"""

import random
from typing import Tuple, List
from .base import BaseTask


class BitwiseAddTask(BaseTask):
    """
    Task: Binary addition of two n-bit numbers.
    
    Format: <binary_a> SEP1 <binary_b> SEP2 <binary_sum> EOS
    LSB first for easier carry propagation learning.
    
    IMPORTANT: Uses proper next-token prediction format:
    - input_seq = full_sequence[:-1]
    - target[i] = full_sequence[i+1] (shifted by 1)
    - Only predict after second SEP token
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_bits = config.num_bits
        
        # Vocab: 0=PAD, 1=SEP, 2=ZERO, 3=ONE, 4=EOS
        self.ZERO_TOKEN = 2
        self.ONE_TOKEN = 3
        self.EOS_TOKEN = 4
        
    def get_vocab_size(self) -> int:
        return 5  # PAD, SEP, 0, 1, EOS
    
    def _int_to_bits(self, n: int, num_bits: int) -> List[int]:
        """Convert integer to list of bits (LSB first)."""
        bits = []
        for _ in range(num_bits):
            bits.append(n & 1)
            n >>= 1
        return bits
    
    def _bits_to_tokens(self, bits: List[int]) -> List[int]:
        """Convert bits to tokens."""
        return [self.ONE_TOKEN if b else self.ZERO_TOKEN for b in bits]
    
    def generate_sample(self, seq_length: int) -> Tuple[List[int], List[int]]:
        """Generate a binary addition sample with proper next-token prediction."""
        num_bits = seq_length  # seq_length determines bit width
        
        # Generate two random n-bit numbers
        max_val = (1 << num_bits) - 1
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        c = a + b  # Result may have num_bits + 1 bits
        
        # Convert to bit sequences (LSB first)
        a_bits = self._int_to_bits(a, num_bits)
        b_bits = self._int_to_bits(b, num_bits)
        c_bits = self._int_to_bits(c, num_bits + 1)  # Extra bit for carry
        
        # Convert to tokens
        a_tokens = self._bits_to_tokens(a_bits)
        b_tokens = self._bits_to_tokens(b_bits)
        c_tokens = self._bits_to_tokens(c_bits)
        
        # Build full sequence: a SEP b SEP c EOS
        full_seq = a_tokens + [self.SEP_TOKEN] + b_tokens + [self.SEP_TOKEN] + c_tokens + [self.EOS_TOKEN]
        
        # Input is everything except last token (EOS)
        input_seq = full_seq[:-1]
        
        # Target is shifted by 1: target[i] = full_seq[i+1]
        # Only predict after the second SEP (output portion)
        target_seq = []
        sep_count = 0
        for i in range(len(input_seq)):
            if input_seq[i] == self.SEP_TOKEN:
                sep_count += 1
            
            if sep_count >= 2:  # After second SEP, predict next token
                target_seq.append(full_seq[i + 1])
            else:
                target_seq.append(-100)  # Ignore input portions
        
        return input_seq, target_seq
    
    def decode_sample(self, tokens: List[int]) -> str:
        """Decode tokens to readable format."""
        parts = []
        for t in tokens:
            if t == self.PAD_TOKEN:
                continue
            elif t == self.SEP_TOKEN:
                parts.append("+")
            elif t == self.ZERO_TOKEN:
                parts.append("0")
            elif t == self.ONE_TOKEN:
                parts.append("1")
        return "".join(parts)
