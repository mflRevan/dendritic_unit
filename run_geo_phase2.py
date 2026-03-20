#!/usr/bin/env python3
"""
GeoField Phase 2: Insertion Point Comparison + Broader Task Coverage
====================================================================
Tests different geo insertion points on sorting, reversal, modular_arith.
Also adds geo_attn_out_scale (Phase 1 winner) to new tasks.
"""

import torch
import json
import time
import os
import sys
import traceback
import torch.nn as nn
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from ablation_suite.config import MODEL_CONFIGS, TASK_CONFIGS, TrainingConfig
from ablation_suite.train import train_model, count_parameters
from ablation_suite.evaluate import Evaluator


# Phase 2 models: best attn_out + other insertion points + baseline
PHASE2_MODELS = [
    "baseline",
    "geo_attn_out_scale",     # Phase 1 winner
    "geo_value_static",       # V projection
    "geo_both_vo_static",     # V + O 
    "geo_mlp_down_static",    # MLP down
    "geo_block_residual",     # block residual branch
]

PHASE2_TASKS = ["sorting", "reversal", "modular_arith"]

RESULTS_FILE = "geofield_phase2_results.json"


def collect_geo_stats(model):
    """Collect diagnostic stats from GeoField modules."""
    stats = {}
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    if hasattr(raw, 'get_geo_stats'):
        raw_stats = raw.get_geo_stats()
        for k, v in raw_stats.items():
            if isinstance(v, torch.Tensor):
                stats[k] = v.detach().float().cpu().tolist()
            else:
                stats[k] = v
    blocks = getattr(raw, 'blocks', [])
    for i, block in enumerate(blocks):
        for attr_name in ['attn', 'mlp', 'geo_branch']:
            mod = getattr(block, attr_name, None)
            if mod is None:
                continue
            for field_name in ['geo_out', 'geo_v', 'geo_field', 'geo_up', 'geo_down']:
                gf = getattr(mod, field_name, None)
                if gf is None:
                    continue
                prefix = f"block{i}_{attr_name}_{field_name}"
                if hasattr(gf, 'angle') and isinstance(gf.angle, nn.Parameter):
                    angle = gf.angle.detach().float()
                    stats[f"{prefix}_angle_mean"] = angle.abs().mean().item()
                    stats[f"{prefix}_angle_max"] = angle.abs().max().item()
                if hasattr(gf, 'axis'):
                    axis = gf.axis.detach().float()
                    stats[f"{prefix}_axis_norm"] = axis.norm(dim=-1).mean().item()
                coords = gf.coords.detach().float()
                stats[f"{prefix}_coord_range"] = (coords.max() - coords.min()).item()
                stats[f"{prefix}_coord_std"] = coords.std().item()
                if hasattr(gf, 'lam'):
                    stats[f"{prefix}_lambda"] = gf.lam.detach().float().item()
                if gf.use_scale and hasattr(gf, 'log_scale'):
                    stats[f"{prefix}_scale"] = gf.log_scale.detach().float().exp().item()
    return stats


def run_experiment(model_name, task_name, train_config, device):
    """Run a single experiment and return results dict."""
    model_config = MODEL_CONFIGS[model_name]
    task_config = TASK_CONFIGS[task_name]
    model, metrics, task = train_model(model_config, task_config, train_config, device)
    evaluator = Evaluator(model, task_config, device, train_config.batch_size)
    eval_results = evaluator.full_evaluation(task, ood_multipliers=[2.0, 3.0, 4.0], ood_samples=1000)
    geo_stats = collect_geo_stats(model) if model_config.arch == "geofield" else {}
    n_params = count_parameters(model)
    best_val_acc = max(metrics.val_accuracies) if metrics.val_accuracies else 0.0

    result = {
        "model": model_name, "task": task_name, "num_params": n_params,
        "best_val_seq_acc": best_val_acc,
        "best_val_token_acc": max(metrics.val_token_accuracies) if metrics.val_token_accuracies else 0.0,
        "final_val_seq_acc": metrics.val_accuracies[-1] if metrics.val_accuracies else 0.0,
        "final_val_token_acc": metrics.val_token_accuracies[-1] if metrics.val_token_accuracies else 0.0,
        "test_seq_acc": eval_results.id_seq_accuracy,
        "test_token_acc": eval_results.id_token_accuracy,
        "ood_2x": eval_results.ood_results.get('2.0x', {}).get('seq_accuracy', 0.0),
        "ood_3x": eval_results.ood_results.get('3.0x', {}).get('seq_accuracy', 0.0),
        "ood_4x": eval_results.ood_results.get('4.0x', {}).get('seq_accuracy', 0.0),
        "train_losses": metrics.train_losses,
        "val_losses": metrics.val_losses,
        "val_acc_history": metrics.val_accuracies,
        "val_token_acc_history": metrics.val_token_accuracies,
        "epoch_times": metrics.epoch_times,
        "geo_stats": geo_stats,
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
                completed.add((r['model'], r['task']))
        print(f"Loaded {len(all_results)} existing results")

    total = len(PHASE2_MODELS) * len(PHASE2_TASKS)
    done = len(completed)

    for model_name in PHASE2_MODELS:
        for task_name in PHASE2_TASKS:
            if (model_name, task_name) in completed:
                print(f"[SKIP] {model_name} x {task_name}")
                continue
            done += 1
            print(f"\n{'='*70}\n[{done}/{total}] {model_name} x {task_name}\n{'='*70}")
            t0 = time.time()
            try:
                result = run_experiment(model_name, task_name, train_config, device)
                all_results.append(result)
                with open(RESULTS_FILE, 'w') as f:
                    json.dump(all_results, f, indent=2)
                print(f"  -> Test Acc: {result['test_seq_acc']*100:.2f}% | Time: {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"  -> FAILED: {e}")
                traceback.print_exc()
                all_results.append({"model": model_name, "task": task_name, "error": str(e), "timestamp": datetime.now().isoformat()})
                with open(RESULTS_FILE, 'w') as f:
                    json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}\nPhase 2 Complete!\n{'='*70}")
    print(f"\n{'Model':<28s} {'Task':<14s} {'Params':>10s} {'TestAcc':>8s} {'OOD2x':>8s}")
    print('-' * 75)
    for r in all_results:
        if 'error' in r:
            print(f"{r['model']:<28s} {r['task']:<14s} ERROR")
            continue
        print(f"{r['model']:<28s} {r['task']:<14s} {r['num_params']:>10,} {r['test_seq_acc']*100:>7.2f}% {r['ood_2x']*100:>7.2f}%")


if __name__ == "__main__":
    main()
