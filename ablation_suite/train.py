"""
Training utilities for the ablation suite.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Optional, Tuple, Any
import time
from dataclasses import dataclass
from tqdm import tqdm

from model.transformer import Transformer
from model.spinformer import Spinformer
from model.geofield_transformer import GeoFieldTransformer
from .config import ModelConfig, TrainingConfig, TaskConfig
from .tasks import get_task


@dataclass
class TrainingMetrics:
    """Metrics accumulated during training."""
    train_losses: List[float]
    val_losses: List[float]
    val_accuracies: List[float]
    val_token_accuracies: List[float]
    learning_rates: List[float]
    epoch_times: List[float]
    

def create_model(model_config: ModelConfig, vocab_size: int, max_seq_len: int) -> torch.nn.Module:
    """Create a model from config (Transformer, Spinformer, or GeoField)."""
    if model_config.arch == "spinformer":
        model = Spinformer(
            vocab_size=vocab_size,
            seq_length=max_seq_len,
            dim=model_config.d_model,
            num_heads=model_config.n_heads,
            num_layers=model_config.n_layers,
            dropout=model_config.dropout,
            num_kv_heads=model_config.num_kv_heads,
            expand_factor=model_config.d_ff // model_config.d_model,
            use_swiglu=model_config.use_swiglu,
            rotation_target=model_config.rotation_target,
            rotation_mode=model_config.rotation_mode,
        )
    elif model_config.arch == "geofield":
        model = GeoFieldTransformer(
            vocab_size=vocab_size,
            seq_length=max_seq_len,
            dim=model_config.d_model,
            num_heads=model_config.n_heads,
            num_layers=model_config.n_layers,
            dropout=model_config.dropout,
            num_kv_heads=model_config.num_kv_heads,
            expand_factor=model_config.d_ff // model_config.d_model,
            use_swiglu=model_config.use_swiglu,
            geo_target=model_config.geo_target,
            geo_mode=model_config.geo_mode,
            geo_conditioning=model_config.geo_conditioning,
            geo_num_coords=model_config.geo_num_coords,
            geo_rank=model_config.geo_rank,
            geo_use_scale=model_config.geo_use_scale,
            geo_use_pivot_offset=model_config.geo_use_pivot_offset,
            geo_num_heads=model_config.geo_num_heads,
            geo_lam_init=model_config.geo_lam_init,
        )
    else:
        model = Transformer(
            vocab_size=vocab_size,
            seq_length=max_seq_len,
            dim=model_config.d_model,
            num_heads=model_config.n_heads,
            num_layers=model_config.n_layers,
            dropout=model_config.dropout,
            num_kv_heads=model_config.num_kv_heads,
            expand_factor=model_config.d_ff // model_config.d_model,
            use_swiglu=model_config.use_swiglu,
        )
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Trainer:
    """Training loop with logging and metrics."""
    
    def __init__(
        self,
        model: nn.Module,
        task,
        train_config: TrainingConfig,
        device: torch.device,
        run_name: str = "run",
    ):
        self.model = model.to(device)
        self.task = task
        self.config = train_config
        self.device = device
        self.run_name = run_name
        
        # Compile model for efficiency
        if train_config.use_compile and device.type == 'cuda':
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode="default")
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            betas=(0.9, 0.95),
        )
        
        # Data loaders
        self.train_loader = task.get_train_loader(train_config.batch_size)
        self.val_loader = task.get_val_loader(train_config.batch_size)
        
        # Scheduler  
        total_steps = len(self.train_loader) * train_config.num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=train_config.learning_rate * 0.01,
        )
        
        # Metrics storage
        self.metrics = TrainingMetrics(
            train_losses=[],
            val_losses=[],
            val_accuracies=[],
            val_token_accuracies=[],
            learning_rates=[],
            epoch_times=[],
        )
        
        # Gradient scaler for AMP
        self.scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
            leave=False,
        )
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.device.type == 'cuda' and self.scaler is not None:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = self.model(inputs)
                    # Flatten for cross entropy
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=-100,
                    )
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(inputs)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100,
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
            
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
            })
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]:
        """Validate and return loss, sequence accuracy, token accuracy."""
        self.model.eval()
        total_loss = 0.0
        total_correct_seqs = 0
        total_seqs = 0
        total_correct_tokens = 0
        total_tokens = 0
        
        for inputs, targets in self.val_loader:
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
            
            total_loss += loss.item()
            
            # Predictions
            preds = logits.argmax(dim=-1)
            
            # Token accuracy (ignoring -100)
            mask = targets != -100
            total_correct_tokens += ((preds == targets) & mask).sum().item()
            total_tokens += mask.sum().item()
            
            # Sequence accuracy
            for i in range(preds.size(0)):
                seq_mask = targets[i] != -100
                if seq_mask.sum() > 0:
                    seq_correct = (preds[i][seq_mask] == targets[i][seq_mask]).all()
                    total_correct_seqs += int(seq_correct)
                    total_seqs += 1
        
        avg_loss = total_loss / len(self.val_loader)
        seq_acc = total_correct_seqs / total_seqs if total_seqs > 0 else 0.0
        token_acc = total_correct_tokens / total_tokens if total_tokens > 0 else 0.0
        
        return avg_loss, seq_acc, token_acc
    
    def train(self) -> TrainingMetrics:
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"Training: {self.run_name}")
        print(f"Model params: {count_parameters(self.model):,}")
        print(f"Train samples: {len(self.task.train_data)}")
        print(f"Val samples: {len(self.task.val_data)}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_token_acc = self.validate()
            
            epoch_time = time.time() - start_time
            lr = self.scheduler.get_last_lr()[0]
            
            # Store metrics
            self.metrics.train_losses.append(train_loss)
            self.metrics.val_losses.append(val_loss)
            self.metrics.val_accuracies.append(val_acc)
            self.metrics.val_token_accuracies.append(val_token_acc)
            self.metrics.learning_rates.append(lr)
            self.metrics.epoch_times.append(epoch_time)
            
            # Track best
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            # Print progress
            print(
                f"Epoch {epoch+1:3d}/{self.config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Seq Acc: {val_acc*100:.2f}% | "
                f"Token Acc: {val_token_acc*100:.2f}% | "
                f"LR: {lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
        
        print(f"\nBest Val Acc: {self.best_val_acc*100:.2f}% at epoch {self.best_epoch+1}")
        return self.metrics


def train_model(
    model_config: ModelConfig,
    task_config: TaskConfig,
    train_config: TrainingConfig,
    device: torch.device,
) -> Tuple[nn.Module, TrainingMetrics]:
    """
    Train a model on a task.
    
    Returns: (trained_model, metrics)
    """
    # Create task
    task = get_task(task_config.name, task_config)
    
    # Create model
    vocab_size = task.get_vocab_size()
    max_seq_len = task_config.max_seq_len * 4  # Allow for OOD testing
    model = create_model(model_config, vocab_size, max_seq_len)
    
    # Create trainer
    run_name = f"{model_config.name}_{task_config.name}"
    trainer = Trainer(model, task, train_config, device, run_name)
    
    # Train
    metrics = trainer.train()
    
    return model, metrics, task
