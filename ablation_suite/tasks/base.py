"""
Base task class for algorithmic tasks.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class AlgorithmicDataset(Dataset):
    """Dataset wrapper for algorithmic tasks."""
    
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs
        self.targets = targets
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class BaseTask(ABC):
    """Base class for algorithmic tasks."""
    
    def __init__(self, config):
        self.config = config
        self.name = config.name
        self.vocab_size = config.vocab_size
        self.train_seq_len = config.train_seq_len
        
        # Special tokens
        self.PAD_TOKEN = 0
        self.SEP_TOKEN = 1
        self.START_TOKEN = 2
        
        # Cache for generated data
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    @abstractmethod
    def generate_sample(self, seq_length: int) -> Tuple[List[int], List[int]]:
        """
        Generate a single (input, target) sample.
        
        Returns:
            input_seq: List of input tokens
            target_seq: List of target tokens (same length, with ignore tokens where needed)
        """
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Return the vocabulary size for this task."""
        pass
    
    def generate_dataset(self, num_samples: int, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a dataset of samples."""
        inputs = []
        targets = []
        
        for _ in range(num_samples):
            inp, tgt = self.generate_sample(seq_length)
            inputs.append(inp)
            targets.append(tgt)
        
        # Pad to max length in batch
        max_len = max(len(x) for x in inputs)
        
        padded_inputs = []
        padded_targets = []
        
        for inp, tgt in zip(inputs, targets):
            pad_len = max_len - len(inp)
            padded_inputs.append(inp + [self.PAD_TOKEN] * pad_len)
            padded_targets.append(tgt + [-100] * pad_len)  # -100 is ignore index
            
        return (
            torch.tensor(padded_inputs, dtype=torch.long),
            torch.tensor(padded_targets, dtype=torch.long)
        )
    
    def get_train_loader(self, batch_size: int) -> DataLoader:
        """Get training data loader."""
        if self.train_data is None:
            inputs, targets = self.generate_dataset(
                self.config.train_samples, 
                self.train_seq_len
            )
            self.train_data = AlgorithmicDataset(inputs, targets)
        
        return DataLoader(
            self.train_data, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
    
    def get_val_loader(self, batch_size: int) -> DataLoader:
        """Get validation data loader (same length as training)."""
        if self.val_data is None:
            inputs, targets = self.generate_dataset(
                self.config.val_samples,
                self.config.val_seq_len
            )
            self.val_data = AlgorithmicDataset(inputs, targets)
        
        return DataLoader(
            self.val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    
    def get_test_loader(self, batch_size: int) -> DataLoader:
        """Get test data loader."""
        if self.test_data is None:
            inputs, targets = self.generate_dataset(
                self.config.test_samples,
                self.config.test_seq_len
            )
            self.test_data = AlgorithmicDataset(inputs, targets)
        
        return DataLoader(
            self.test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    
    def compute_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute accuracy metrics.
        
        Args:
            predictions: [batch, seq_len] predicted tokens
            targets: [batch, seq_len] target tokens (-100 for ignore)
        
        Returns:
            Dictionary with token_accuracy and sequence_accuracy
        """
        # Mask for valid positions
        mask = targets != -100
        
        if mask.sum() == 0:
            return {"token_accuracy": 0.0, "sequence_accuracy": 0.0}
        
        # Token-level accuracy
        correct_tokens = ((predictions == targets) & mask).sum().item()
        total_tokens = mask.sum().item()
        token_accuracy = correct_tokens / total_tokens
        
        # Sequence-level accuracy (all tokens correct)
        seq_correct = ((predictions == targets) | ~mask).all(dim=1)
        sequence_accuracy = seq_correct.float().mean().item()
        
        return {
            "token_accuracy": token_accuracy,
            "sequence_accuracy": sequence_accuracy
        }
    
    def decode_sample(self, tokens: List[int]) -> str:
        """Decode tokens to human-readable string (for debugging)."""
        return " ".join(str(t) for t in tokens if t != self.PAD_TOKEN)
