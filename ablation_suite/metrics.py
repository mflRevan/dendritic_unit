"""
Metrics tracking and plotting utilities.
"""

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunResult:
    """Results from a single run."""
    model_name: str
    task_name: str
    
    # Training metrics
    final_train_loss: float
    final_val_loss: float
    final_val_seq_acc: float
    final_val_token_acc: float
    best_val_seq_acc: float
    best_epoch: int
    
    # Test metrics
    test_seq_acc: float
    test_token_acc: float
    
    # OOD metrics
    ood_2x_seq_acc: float = 0.0
    ood_3x_seq_acc: float = 0.0
    ood_4x_seq_acc: float = 0.0
    
    # Metadata
    num_params: int = 0
    total_time: float = 0.0
    timestamp: str = ""
    
    # Full history (optional)
    train_loss_history: List[float] = None
    val_loss_history: List[float] = None
    val_acc_history: List[float] = None


class MetricsTracker:
    """Track and aggregate results across runs."""
    
    def __init__(self, save_dir: str = "results"):
        self.save_dir = save_dir
        self.results: List[RunResult] = []
        os.makedirs(save_dir, exist_ok=True)
        
    def add_result(self, result: RunResult):
        """Add a result."""
        self.results.append(result)
        
    def save(self, filename: str = "results.json"):
        """Save results to JSON."""
        path = os.path.join(self.save_dir, filename)
        data = [asdict(r) for r in self.results]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Saved results to {path}")
        
    def load(self, filename: str = "results.json"):
        """Load results from JSON."""
        path = os.path.join(self.save_dir, filename)
        with open(path, 'r') as f:
            data = json.load(f)
        self.results = [RunResult(**d) for d in data]
        print(f"Loaded {len(self.results)} results from {path}")
        
    def get_summary_table(self) -> str:
        """Generate a summary table."""
        # Group by task
        tasks = sorted(set(r.task_name for r in self.results))
        models = sorted(set(r.model_name for r in self.results))
        
        lines = []
        lines.append("="*100)
        lines.append("RESULTS SUMMARY")
        lines.append("="*100)
        
        for task in tasks:
            lines.append(f"\n--- {task.upper()} ---")
            lines.append(f"{'Model':<35} {'Test Acc':>10} {'OOD 2x':>10} {'OOD 3x':>10} {'OOD 4x':>10}")
            lines.append("-"*75)
            
            task_results = [r for r in self.results if r.task_name == task]
            for model in models:
                model_results = [r for r in task_results if r.model_name == model]
                if model_results:
                    r = model_results[0]  # Take first if multiple
                    lines.append(
                        f"{r.model_name:<35} "
                        f"{r.test_seq_acc*100:>9.2f}% "
                        f"{r.ood_2x_seq_acc*100:>9.2f}% "
                        f"{r.ood_3x_seq_acc*100:>9.2f}% "
                        f"{r.ood_4x_seq_acc*100:>9.2f}%"
                    )
        
        return "\n".join(lines)
    
    def print_summary(self):
        """Print summary table."""
        print(self.get_summary_table())
        
    def plot_training_curves(self, save: bool = True):
        """Plot training curves for all runs."""
        tasks = sorted(set(r.task_name for r in self.results))
        
        for task in tasks:
            task_results = [r for r in self.results if r.task_name == task and r.val_acc_history]
            if not task_results:
                continue
                
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Loss curves
            ax1 = axes[0]
            for r in task_results:
                if r.train_loss_history:
                    ax1.plot(r.train_loss_history, label=f"{r.model_name} (train)", linestyle='--', alpha=0.7)
                if r.val_loss_history:
                    ax1.plot(r.val_loss_history, label=f"{r.model_name} (val)")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title(f"{task} - Training Loss")
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Accuracy curves
            ax2 = axes[1]
            for r in task_results:
                if r.val_acc_history:
                    ax2.plot(r.val_acc_history, label=r.model_name)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Sequence Accuracy")
            ax2.set_title(f"{task} - Validation Accuracy")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.05)
            
            plt.tight_layout()
            
            if save:
                path = os.path.join(self.save_dir, f"training_{task}.png")
                plt.savefig(path, dpi=150)
                print(f"Saved training curves to {path}")
            plt.close()
    
    def plot_ood_comparison(self, save: bool = True):
        """Plot OOD generalization comparison."""
        tasks = sorted(set(r.task_name for r in self.results))
        models = sorted(set(r.model_name for r in self.results))
        
        for task in tasks:
            task_results = [r for r in self.results if r.task_name == task]
            if not task_results:
                continue
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(4)  # ID, 2x, 3x, 4x
            width = 0.15
            
            for i, model in enumerate(models):
                model_results = [r for r in task_results if r.model_name == model]
                if model_results:
                    r = model_results[0]
                    values = [r.test_seq_acc, r.ood_2x_seq_acc, r.ood_3x_seq_acc, r.ood_4x_seq_acc]
                    offset = (i - len(models)/2 + 0.5) * width
                    bars = ax.bar(x + offset, [v*100 for v in values], width, label=model)
            
            ax.set_xlabel("Sequence Length")
            ax.set_ylabel("Sequence Accuracy (%)")
            ax.set_title(f"{task} - Length Generalization")
            ax.set_xticks(x)
            ax.set_xticklabels(['ID (1x)', 'OOD 2x', 'OOD 3x', 'OOD 4x'])
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 105)
            
            plt.tight_layout()
            
            if save:
                path = os.path.join(self.save_dir, f"ood_{task}.png")
                plt.savefig(path, dpi=150)
                print(f"Saved OOD comparison to {path}")
            plt.close()
    
    def plot_summary_heatmap(self, save: bool = True):
        """Plot heatmap of all results."""
        tasks = sorted(set(r.task_name for r in self.results))
        models = sorted(set(r.model_name for r in self.results))
        
        if not tasks or not models:
            return
            
        # ID accuracy heatmap
        data = np.zeros((len(models), len(tasks)))
        
        for i, model in enumerate(models):
            for j, task in enumerate(tasks):
                results = [r for r in self.results if r.model_name == model and r.task_name == task]
                if results:
                    data[i, j] = results[0].test_seq_acc * 100
        
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        ax.set_xticks(np.arange(len(tasks)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(tasks, rotation=45, ha='right')
        ax.set_yticklabels(models)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(tasks)):
                text = ax.text(j, i, f"{data[i, j]:.1f}%",
                              ha="center", va="center", 
                              color="black" if 30 < data[i, j] < 70 else "white",
                              fontsize=9)
        
        ax.set_title("Test Sequence Accuracy (%) by Model and Task")
        fig.colorbar(im, ax=ax, label="Accuracy (%)")
        
        plt.tight_layout()
        
        if save:
            path = os.path.join(self.save_dir, "summary_heatmap.png")
            plt.savefig(path, dpi=150)
            print(f"Saved summary heatmap to {path}")
        plt.close()
        
    def generate_all_plots(self):
        """Generate all plots."""
        self.plot_training_curves()
        self.plot_ood_comparison()
        self.plot_summary_heatmap()


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"
