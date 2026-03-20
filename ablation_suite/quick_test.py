#!/usr/bin/env python3
"""
Quick test script for the ablation suite.
Run a single task-model combination for verification.
"""

import torch
import argparse

from ablation_suite.config import MODEL_CONFIGS, TASK_CONFIGS, TrainingConfig
from ablation_suite.train import train_model
from ablation_suite.evaluate import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='sorting', choices=list(TASK_CONFIGS.keys()))
    parser.add_argument('--model', default='baseline', choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    train_config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )
    
    model_config = MODEL_CONFIGS[args.model]
    task_config = TASK_CONFIGS[args.task]
    
    print(f"\nTraining {model_config.name} on {args.task}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    
    # Train
    model, metrics, task = train_model(model_config, task_config, train_config, device)
    
    # Evaluate with OOD
    print("\n" + "="*50)
    print("OOD Evaluation")
    print("="*50)
    
    evaluator = Evaluator(model, task_config, device, args.batch_size)
    results = evaluator.full_evaluation(task, ood_multipliers=[2.0, 3.0], ood_samples=500)
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Test Accuracy: {results.id_seq_accuracy*100:.2f}%")
    for mult, res in results.ood_results.items():
        print(f"OOD {mult}: {res['seq_accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()
