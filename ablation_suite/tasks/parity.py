"""
Parity/XOR Chain Task: Compute XOR of a sequence of bits.

Input: b[0] b[1] ... b[n-1] SEP
Output: result (single bit)

Tests sequential XOR propagation.
Simple but requires tracking state through entire sequence.
"""

import random
from typing import Tuple, List
from functools import reduce
from .base import BaseTask


class ParityTask(BaseTask):
    """
    Task: Compute parity (XOR) of a sequence of bits.
    
    Format: <bits> SEP <parity> EOS
    
    IMPORTANT: Uses proper next-token prediction format:
    - input_seq = full_sequence[:-1]
    - target[i] = full_sequence[i+1] (shifted by 1)
    - Only predict after SEP token
    
    This is challenging because:
    1. Any single bit flip changes the answer
    2. Must process entire sequence correctly
    3. No shortcuts - must aggregate information from all positions
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Vocab: 0=PAD, 1=SEP, 2=ZERO, 3=ONE, 4=EOS
        self.ZERO_TOKEN = 2
        self.ONE_TOKEN = 3
        self.EOS_TOKEN = 4
        
    def get_vocab_size(self) -> int:
        return 5  # PAD, SEP, 0, 1, EOS
    
    def generate_sample(self, seq_length: int) -> Tuple[List[int], List[int]]:
        """Generate a parity computation sample with proper next-token prediction."""
        # Generate random bits
        bits = [random.randint(0, 1) for _ in range(seq_length)]
        
        # Compute XOR
        parity = reduce(lambda x, y: x ^ y, bits, 0)
        
        # Convert to tokens
        bit_tokens = [self.ONE_TOKEN if b else self.ZERO_TOKEN for b in bits]
        parity_token = self.ONE_TOKEN if parity else self.ZERO_TOKEN
        
        # Build full sequence: bits SEP parity EOS
        full_seq = bit_tokens + [self.SEP_TOKEN] + [parity_token] + [self.EOS_TOKEN]
        
        # Input is everything except last token (EOS)
        input_seq = full_seq[:-1]
        
        # Target is shifted by 1: target[i] = full_seq[i+1]
        # Only predict after SEP (output portion)
        target_seq = []
        sep_seen = False
        for i in range(len(input_seq)):
            if input_seq[i] == self.SEP_TOKEN:
                sep_seen = True
            
            if sep_seen:  # After SEP, predict next token
                target_seq.append(full_seq[i + 1])
            else:
                target_seq.append(-100)  # Ignore input bits
        
        return input_seq, target_seq
    
    def decode_sample(self, tokens: List[int]) -> str:
        """Decode tokens to readable format."""
        parts = []
        for t in tokens:
            if t == self.PAD_TOKEN:
                continue
            elif t == self.SEP_TOKEN:
                parts.append("→")
            elif t == self.ZERO_TOKEN:
                parts.append("0")
            elif t == self.ONE_TOKEN:
                parts.append("1")
        return "".join(parts)


class XORChainTask(BaseTask):
    """
    Alternative XOR task: Running XOR chain.
    
    Format: b[0] b[1] ... b[n-1] SEP r[0] r[1] ... r[n-1] EOS
    where r[i] = b[0] XOR b[1] XOR ... XOR b[i]
    
    IMPORTANT: Uses proper next-token prediction format:
    - input_seq = full_sequence[:-1]
    - target[i] = full_sequence[i+1] (shifted by 1)
    - Only predict after SEP token
    
    This tests cumulative XOR propagation through sequence.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Vocab: 0=PAD, 1=SEP, 2=ZERO, 3=ONE, 4=EOS
        self.ZERO_TOKEN = 2
        self.ONE_TOKEN = 3
        self.EOS_TOKEN = 4
        
    def get_vocab_size(self) -> int:
        return 5  # PAD, SEP, 0, 1, EOS
    
    def generate_sample(self, seq_length: int) -> Tuple[List[int], List[int]]:
        """Generate a running XOR chain sample with proper next-token prediction."""
        # Generate random bits
        bits = [random.randint(0, 1) for _ in range(seq_length)]
        
        # Compute running XOR
        running_xor = []
        acc = 0
        for b in bits:
            acc ^= b
            running_xor.append(acc)
        
        # Convert to tokens
        bit_tokens = [self.ONE_TOKEN if b else self.ZERO_TOKEN for b in bits]
        result_tokens = [self.ONE_TOKEN if r else self.ZERO_TOKEN for r in running_xor]
        
        # Build full sequence: bits SEP running_xor EOS
        full_seq = bit_tokens + [self.SEP_TOKEN] + result_tokens + [self.EOS_TOKEN]
        
        # Input is everything except last token (EOS)
        input_seq = full_seq[:-1]
        
        # Target is shifted by 1: target[i] = full_seq[i+1]
        # Only predict after SEP (output portion)
        target_seq = []
        sep_seen = False
        for i in range(len(input_seq)):
            if input_seq[i] == self.SEP_TOKEN:
                sep_seen = True
            
            if sep_seen:  # After SEP, predict next token
                target_seq.append(full_seq[i + 1])
            else:
                target_seq.append(-100)  # Ignore input bits
        
        return input_seq, target_seq
    
    def decode_sample(self, tokens: List[int]) -> str:
        """Decode tokens to readable format."""
        parts = []
        for t in tokens:
            if t == self.PAD_TOKEN:
                continue
            elif t == self.SEP_TOKEN:
                parts.append("→")
            elif t == self.ZERO_TOKEN:
                parts.append("0")
            elif t == self.ONE_TOKEN:
                parts.append("1")
        return "".join(parts)
