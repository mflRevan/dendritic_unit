#!/usr/bin/env python3
"""
GeoField Phase 3: Multi-Seed Stability Validation
===================================================
Runs 5 seeds for the top 4 variants + baseline on sorting.
Goal: determine if the Phase 1/2 improvements are stable or seed-dependent.
"""

import torch
import json
import time
import os
import sys
import traceback
import torch.nn as nn
import random
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from ablation_suite.config import MODEL_CONFIGS, TASK_CONFIGS, TrainingConfig
from ablation_suite.train import train_model, count_parameters
from ablation_suite.evaluate import Evaluator


MODELS = [
    "baseline",
    "geo_attn_out_scale",     # Phase 1 winner
    "geo_attn_out_perhead",   # Best params/accuracy
    "geo_value_static",       # Value insertion
    "geo_both_vo_static",     # Phase 2 winner
]

SEEDS = [42, 123, 456, 789, 2024]
TASK = "sorting"
RESULTS_FILE = "geofield_phase3_results.json"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_experiment(model_name, task_name, seed, train_config, device):
    """Run seeded experiment."""
    set_seed(seed)
    model_config = MODEL_CONFIGS[model_name]
    task_config = TASK_CONFIGS[task_name]
    model, metrics, task = train_model(model_config, task_config, train_config, device)
    evaluator = Evaluator(model, task_config, device, train_config.batch_size)
    eval_results = evaluator.full_evaluation(task, ood_multipliers=[2.0], ood_samples=1000)
    n_params = count_parameters(model)

    result = {
        "model": model_name, "task": task_name, "seed": seed,
        "num_params": n_params,
        "best_val_seq_acc": max(metrics.val_accuracies) if metrics.val_accuracies else 0.0,
        "test_seq_acc": eval_results.id_seq_accuracy,
        "test_token_acc": eval_results.id_token_accuracy,
        "ood_2x": eval_results.ood_results.get('2.0x', {}).get('seq_accuracy', 0.0),
        "val_acc_history": metrics.val_accuracies,
        "epoch_times": metrics.epoch_times,
        "timestamp": datetime.now().isoformat(),
    }
    del model; torch.cuda.empty_cache()
    return result


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_config = TrainingConfig(
        num_epochs=10, batch_size=8, learning_rate=1e-3,
        weight_decay=0.1, grad_clip=1.0, use_compile=True,
    )

    all_results = []
    completed = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            all_results = json.load(f)
        for r in all_results:
            if 'error' not in r:
                completed.add((r['model'], r['seed']))

    total = len(MODELS) * len(SEEDS)
    done = len(completed)

    for model_name in MODELS:
        for seed in SEEDS:
            if (model_name, seed) in completed:
                print(f"[SKIP] {model_name} seed={seed}")
                continue
            done += 1
            print(f"\n{'='*60}\n[{done}/{total}] {model_name} seed={seed}\n{'='*60}")
            t0 = time.time()
            try:
                result = run_experiment(model_name, TASK, seed, train_config, device)
                all_results.append(result)
                with open(RESULTS_FILE, 'w') as f:
                    json.dump(all_results, f, indent=2)
                print(f"  -> Test Acc: {result['test_seq_acc']*100:.2f}% | Time: {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"  -> FAILED: {e}")
                traceback.print_exc()

    # Summary
    print(f"\n{'='*70}\nPhase 3 Multi-Seed Summary\n{'='*70}")
    from collections import defaultdict
    model_accs = defaultdict(list)
    for r in all_results:
        if 'error' not in r:
            model_accs[r['model']].append(r['test_seq_acc'])

    print(f"{'Model':<28s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'N':>3s}")
    print('-' * 60)
    for m in MODELS:
        accs = model_accs.get(m, [])
        if accs:
            mean = sum(accs)/len(accs) * 100
            std = (sum((a*100 - mean)**2 for a in accs) / len(accs)) ** 0.5
            print(f"{m:<28s} {mean:>7.1f}% {std:>7.1f}% {min(accs)*100:>7.1f}% {max(accs)*100:>7.1f}% {len(accs):>3d}")


if __name__ == "__main__":
    main()
