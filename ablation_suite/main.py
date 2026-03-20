#!/usr/bin/env python3
"""
Main runner for algorithmic ablation suite.

Usage:
    python -m ablation_suite.main --tasks sorting reversal --models baseline
    python -m ablation_suite.main --all  # Run all tasks and models
"""

import argparse
import torch
import time
import os
from datetime import datetime

from ablation_suite.config import (
    get_model_configs, 
    get_task_configs, 
    TrainingConfig,
    MODEL_CONFIGS,
    TASK_CONFIGS,
)
from ablation_suite.train import train_model, count_parameters
from ablation_suite.evaluate import Evaluator
from ablation_suite.metrics import MetricsTracker, RunResult, format_time


def parse_args():
    parser = argparse.ArgumentParser(description="Algorithmic Ablation Suite")
    
    # Task selection
    parser.add_argument(
        '--tasks',
        nargs='+',
        choices=list(TASK_CONFIGS.keys()),
        default=None,
        help='Tasks to run. Default: all tasks'
    )
    
    # Model selection
    parser.add_argument(
        '--models',
        nargs='+',
        choices=list(MODEL_CONFIGS.keys()),
        default=None,
        help='Models to run. Default: all models'
    )
    
    # Run all
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all tasks and models'
    )
    
    # Training config
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile')
    
    # OOD evaluation
    parser.add_argument(
        '--ood-multipliers',
        nargs='+',
        type=float,
        default=[2.0, 3.0, 4.0],
        help='OOD length multipliers'
    )
    parser.add_argument('--ood-samples', type=int, default=1000, help='Samples per OOD evaluation')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='ablation_results', help='Output directory')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    
    # Device
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/cpu)')
    
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Get the appropriate device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    return torch.device(device_arg)


def run_single_experiment(
    model_name: str,
    task_name: str,
    train_config: TrainingConfig,
    device: torch.device,
    ood_multipliers: list,
    ood_samples: int,
) -> RunResult:
    """Run a single model-task experiment."""
    
    print("\n" + "="*80)
    print(f"EXPERIMENT: {model_name} on {task_name}")
    print("="*80)
    
    start_time = time.time()
    
    # Get configs
    model_config = MODEL_CONFIGS[model_name]
    task_config = TASK_CONFIGS[task_name]
    
    # Train
    model, metrics, task = train_model(model_config, task_config, train_config, device)
    
    # Evaluate
    evaluator = Evaluator(model, task_config, device, train_config.batch_size)
    eval_results = evaluator.full_evaluation(
        task,
        ood_multipliers=ood_multipliers,
        ood_samples=ood_samples,
    )
    
    total_time = time.time() - start_time
    
    # Extract OOD results
    ood_2x = eval_results.ood_results.get('2.0x', {}).get('seq_accuracy', 0.0)
    ood_3x = eval_results.ood_results.get('3.0x', {}).get('seq_accuracy', 0.0)
    ood_4x = eval_results.ood_results.get('4.0x', {}).get('seq_accuracy', 0.0)
    
    # Find best epoch
    best_val_acc = max(metrics.val_accuracies) if metrics.val_accuracies else 0.0
    best_epoch = metrics.val_accuracies.index(best_val_acc) if metrics.val_accuracies else 0
    
    result = RunResult(
        model_name=model_name,
        task_name=task_name,
        final_train_loss=metrics.train_losses[-1] if metrics.train_losses else 0.0,
        final_val_loss=metrics.val_losses[-1] if metrics.val_losses else 0.0,
        final_val_seq_acc=metrics.val_accuracies[-1] if metrics.val_accuracies else 0.0,
        final_val_token_acc=metrics.val_token_accuracies[-1] if metrics.val_token_accuracies else 0.0,
        best_val_seq_acc=best_val_acc,
        best_epoch=best_epoch,
        test_seq_acc=eval_results.id_seq_accuracy,
        test_token_acc=eval_results.id_token_accuracy,
        ood_2x_seq_acc=ood_2x,
        ood_3x_seq_acc=ood_3x,
        ood_4x_seq_acc=ood_4x,
        num_params=count_parameters(model),
        total_time=total_time,
        timestamp=datetime.now().isoformat(),
        train_loss_history=metrics.train_losses,
        val_loss_history=metrics.val_losses,
        val_acc_history=metrics.val_accuracies,
    )
    
    print(f"\nExperiment completed in {format_time(total_time)}")
    print(f"Test Accuracy: {result.test_seq_acc*100:.2f}%")
    print(f"OOD 2x: {ood_2x*100:.2f}% | OOD 3x: {ood_3x*100:.2f}% | OOD 4x: {ood_4x*100:.2f}%")
    
    return result


def main():
    args = parse_args()
    
    # Determine tasks and models
    if args.all or (args.tasks is None and args.models is None):
        tasks = list(TASK_CONFIGS.keys())
        models = list(MODEL_CONFIGS.keys())
    else:
        tasks = args.tasks or list(TASK_CONFIGS.keys())
        models = args.models or list(MODEL_CONFIGS.keys())
    
    # Setup
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Training config
    train_config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_compile=not args.no_compile,
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Metrics tracker
    tracker = MetricsTracker(save_dir=args.output_dir)
    
    # Run experiments
    total_experiments = len(tasks) * len(models)
    experiment_num = 0
    
    print("\n" + "#"*80)
    print(f"ABLATION SUITE")
    print(f"Tasks: {tasks}")
    print(f"Models: {models}")
    print(f"Total experiments: {total_experiments}")
    print("#"*80)
    
    suite_start = time.time()
    
    for task_name in tasks:
        for model_name in models:
            experiment_num += 1
            print(f"\n[{experiment_num}/{total_experiments}]")
            
            try:
                result = run_single_experiment(
                    model_name=model_name,
                    task_name=task_name,
                    train_config=train_config,
                    device=device,
                    ood_multipliers=args.ood_multipliers,
                    ood_samples=args.ood_samples,
                )
                tracker.add_result(result)
                
                # Save incrementally
                tracker.save()
                
            except Exception as e:
                print(f"ERROR in {model_name} on {task_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    suite_time = time.time() - suite_start
    
    # Final summary
    print("\n" + "#"*80)
    print("FINAL SUMMARY")
    print("#"*80)
    print(f"\nTotal time: {format_time(suite_time)}")
    tracker.print_summary()
    
    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        tracker.generate_all_plots()
    
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

