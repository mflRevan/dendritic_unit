"""
Evaluation utilities with OOD length generalization testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import time
from tqdm import tqdm

from .tasks import get_task
from .config import TaskConfig


@dataclass
class EvalResults:
    """Results from evaluation."""
    # In-distribution metrics
    id_loss: float
    id_seq_accuracy: float
    id_token_accuracy: float
    
    # Out-of-distribution metrics by length multiplier
    ood_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Per-sample predictions (for analysis)
    sample_predictions: List[Dict] = field(default_factory=list)


class Evaluator:
    """Evaluation with OOD length generalization testing."""
    
    def __init__(
        self,
        model: nn.Module,
        task_config: TaskConfig,
        device: torch.device,
        batch_size: int = 64,
    ):
        self.model = model
        self.task_config = task_config
        self.device = device
        self.batch_size = batch_size
        
    @torch.no_grad()
    def evaluate_loader(
        self,
        loader,
        desc: str = "Evaluating",
    ) -> Tuple[float, float, float, List[Dict]]:
        """
        Evaluate on a dataloader.
        
        Returns: (loss, seq_accuracy, token_accuracy, sample_predictions)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct_seqs = 0
        total_seqs = 0
        total_correct_tokens = 0
        total_tokens = 0
        sample_predictions = []
        
        for inputs, targets in tqdm(loader, desc=desc, leave=False):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            if self.device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = self.model(inputs)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=-100,
                    )
            else:
                logits = self.model(inputs)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100,
                )
            
            total_loss += loss.item() * inputs.size(0)
            
            # Predictions
            preds = logits.argmax(dim=-1)
            
            # Token accuracy
            mask = targets != -100
            total_correct_tokens += ((preds == targets) & mask).sum().item()
            total_tokens += mask.sum().item()
            
            # Sequence accuracy and sample tracking
            for i in range(preds.size(0)):
                seq_mask = targets[i] != -100
                if seq_mask.sum() > 0:
                    pred_seq = preds[i][seq_mask].cpu().tolist()
                    target_seq = targets[i][seq_mask].cpu().tolist()
                    seq_correct = pred_seq == target_seq
                    total_correct_seqs += int(seq_correct)
                    total_seqs += 1
                    
                    # Store sample for analysis
                    if len(sample_predictions) < 100:  # Limit storage
                        sample_predictions.append({
                            'input': inputs[i].cpu().tolist(),
                            'target': target_seq,
                            'prediction': pred_seq,
                            'correct': seq_correct,
                        })
        
        num_samples = total_seqs
        avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
        seq_acc = total_correct_seqs / total_seqs if total_seqs > 0 else 0.0
        token_acc = total_correct_tokens / total_tokens if total_tokens > 0 else 0.0
        
        return avg_loss, seq_acc, token_acc, sample_predictions
    
    def evaluate_id(self, task) -> Tuple[float, float, float, List[Dict]]:
        """Evaluate in-distribution (test set)."""
        loader = task.get_test_loader(self.batch_size)
        return self.evaluate_loader(loader, desc="ID Evaluation")
    
    def evaluate_ood(
        self,
        length_multiplier: float,
        num_samples: int = 1000,
    ) -> Tuple[float, float, float, List[Dict]]:
        """
        Evaluate out-of-distribution at a different length.
        
        Args:
            length_multiplier: Multiply training length by this factor
            num_samples: Number of samples to generate
        """
        # Create a task config for OOD length
        ood_seq_len = int(self.task_config.train_seq_len * length_multiplier)
        
        # Create modified config
        ood_config = TaskConfig(
            name=self.task_config.name,
            train_seq_len=ood_seq_len,
            val_seq_len=ood_seq_len,
            test_seq_len=ood_seq_len,
            train_samples=num_samples,
            val_samples=num_samples,
            test_samples=num_samples,
            max_seq_len=self.task_config.max_seq_len * int(length_multiplier + 1),
            vocab_size=self.task_config.vocab_size,
            num_bits=int(self.task_config.num_bits * length_multiplier) if self.task_config.num_bits else None,
        )
        
        # Create task and generate data
        ood_task = get_task(self.task_config.name, ood_config)
        
        loader = ood_task.get_test_loader(self.batch_size)
        return self.evaluate_loader(loader, desc=f"OOD {length_multiplier}x")
    
    def full_evaluation(
        self,
        task,
        ood_multipliers: List[float] = [2.0, 3.0, 4.0],
        ood_samples: int = 1000,
    ) -> EvalResults:
        """
        Full evaluation including ID and OOD.
        
        Args:
            task: The task instance used for training
            ood_multipliers: Length multipliers for OOD testing
            ood_samples: Number of samples per OOD evaluation
        """
        print("\n" + "="*50)
        print("Full Evaluation")
        print("="*50)
        
        # In-distribution evaluation
        print("\nIn-Distribution Test:")
        id_loss, id_seq_acc, id_token_acc, id_samples = self.evaluate_id(task)
        print(f"  Loss: {id_loss:.4f}")
        print(f"  Seq Accuracy: {id_seq_acc*100:.2f}%")
        print(f"  Token Accuracy: {id_token_acc*100:.2f}%")
        
        # OOD evaluations
        ood_results = {}
        for mult in ood_multipliers:
            print(f"\nOOD {mult}x Length:")
            try:
                ood_loss, ood_seq_acc, ood_token_acc, _ = self.evaluate_ood(mult, ood_samples)
                ood_results[f"{mult}x"] = {
                    'loss': ood_loss,
                    'seq_accuracy': ood_seq_acc,
                    'token_accuracy': ood_token_acc,
                }
                print(f"  Loss: {ood_loss:.4f}")
                print(f"  Seq Accuracy: {ood_seq_acc*100:.2f}%")
                print(f"  Token Accuracy: {ood_token_acc*100:.2f}%")
            except Exception as e:
                print(f"  Error: {e}")
                ood_results[f"{mult}x"] = {
                    'loss': float('inf'),
                    'seq_accuracy': 0.0,
                    'token_accuracy': 0.0,
                    'error': str(e),
                }
        
        return EvalResults(
            id_loss=id_loss,
            id_seq_accuracy=id_seq_acc,
            id_token_accuracy=id_token_acc,
            ood_results=ood_results,
            sample_predictions=id_samples,
        )


def quick_eval(
    model: nn.Module,
    task,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Quick evaluation returning just key metrics."""
    model.eval()
    loader = task.get_test_loader(batch_size)
    
    total_correct_seqs = 0
    total_seqs = 0
    total_correct_tokens = 0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            if device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(inputs)
            else:
                logits = model(inputs)
            
            preds = logits.argmax(dim=-1)
            
            mask = targets != -100
            total_correct_tokens += ((preds == targets) & mask).sum().item()
            total_tokens += mask.sum().item()
            
            for i in range(preds.size(0)):
                seq_mask = targets[i] != -100
                if seq_mask.sum() > 0:
                    seq_correct = (preds[i][seq_mask] == targets[i][seq_mask]).all()
                    total_correct_seqs += int(seq_correct)
                    total_seqs += 1
    
    return {
        'seq_accuracy': total_correct_seqs / total_seqs if total_seqs > 0 else 0.0,
        'token_accuracy': total_correct_tokens / total_tokens if total_tokens > 0 else 0.0,
    }
