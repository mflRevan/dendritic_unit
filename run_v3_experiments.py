"""
Experiment v3: Complete coverage for spin_gated_mlp.

Run gated, local_mlp, and baseline on the 3 remaining tasks (reversal, 
modular_arith, parity) to complete the profile of the best variant.
Also run 3 seeds of spin_global_mlp on bitwise_add to quantify instability.
"""

import torch
import torch.nn as nn
import time
import json
import os

from ablation_suite.config import MODEL_CONFIGS, TASK_CONFIGS, TrainingConfig
from ablation_suite.train import create_model, Trainer, count_parameters, train_model


def run_single_experiment(model_name, task_name, device, train_config, seed=None):
    """Run a single (model, task) experiment and return results dict."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    model_cfg = MODEL_CONFIGS[model_name]
    task_cfg = TASK_CONFIGS[task_name]
    effective_config = train_config.for_task(task_name)

    print(f"\n{'#'*70}")
    print(f"# {model_name} x {task_name}" + (f" (seed={seed})" if seed is not None else ""))
    print(f"{'#'*70}")

    start = time.time()
    model, metrics, task = train_model(model_cfg, task_cfg, effective_config, device)
    wall_time = time.time() - start

    n_params = count_parameters(model)

    # Gather spin stats
    spin_stats = {}
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    if hasattr(raw_model, 'get_spin_stats'):
        spin_stats = raw_model.get_spin_stats()

    # Gradient norms
    grad_norms = {}
    raw_model.eval()
    raw_model.train()
    sample_x = torch.randint(0, task.get_vocab_size(), (4, task_cfg.train_seq_len * 2), device=device)
    sample_y = sample_x.clone()
    sample_y[:, :task_cfg.train_seq_len] = -100

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = raw_model(sample_x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            sample_y.view(-1),
            ignore_index=-100,
        )
    loss.backward()

    for name, param in raw_model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    result = {
        "model": model_name,
        "task": task_name,
        "seed": seed,
        "arch": model_cfg.arch,
        "n_params": n_params,
        "wall_time_s": wall_time,
        "best_val_seq_acc": max(metrics.val_accuracies) if metrics.val_accuracies else 0.0,
        "best_val_token_acc": max(metrics.val_token_accuracies) if metrics.val_token_accuracies else 0.0,
        "final_val_seq_acc": metrics.val_accuracies[-1] if metrics.val_accuracies else 0.0,
        "final_train_loss": metrics.train_losses[-1] if metrics.train_losses else float("inf"),
        "val_seq_accs": metrics.val_accuracies,
        "train_losses": metrics.train_losses,
        "epoch_times": metrics.epoch_times,
        "spin_stats": {k: {sk: sv for sk, sv in v.items() if 'weight_norm' in sk or 'gate' in sk} for k, v in spin_stats.items()} if spin_stats else {},
        "grad_norm_summary": {
            "mean": sum(grad_norms.values()) / len(grad_norms) if grad_norms else 0,
            "max": max(grad_norms.values()) if grad_norms else 0,
            "min": min(grad_norms.values()) if grad_norms else 0,
        },
    }

    print(f"\n>>> {model_name} x {task_name}: "
          f"Best Seq Acc = {result['best_val_seq_acc']*100:.2f}%, "
          f"Time = {wall_time:.1f}s")

    return result


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    train_config = TrainingConfig(
        num_epochs=10,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=0.1,
        grad_clip=1.0,
        use_compile=True,
        task_overrides={
            "parity": {"num_epochs": 20},
        },
    )

    # Part 1: Complete task coverage for the top 3 variants
    remaining_tasks = ["reversal", "modular_arith", "parity"]
    coverage_models = ["baseline", "spin_local_mlp", "spin_gated_mlp"]

    # Part 2: Multi-seed stability test on bitwise_add
    stability_models = ["spin_global_mlp", "spin_gated_mlp"]
    stability_seeds = [42, 123, 7]

    results_file = "ablation_v3_results.json"
    all_results = []
    completed_keys = set()
    
    if os.path.exists(results_file):
        with open(results_file) as f:
            all_results = json.load(f)
        completed_keys = {(r["model"], r["task"], r.get("seed")) for r in all_results}
        if completed_keys:
            print(f"Resuming: {len(completed_keys)} experiments already done")

    # Build experiment list
    experiments = []
    for task_name in remaining_tasks:
        for model_name in coverage_models:
            experiments.append((model_name, task_name, None))
    
    for model_name in stability_models:
        for seed in stability_seeds:
            experiments.append((model_name, "bitwise_add", seed))

    total = len(experiments)
    
    for idx, (model_name, task_name, seed) in enumerate(experiments, 1):
        if (model_name, task_name, seed) in completed_keys:
            print(f"\n[{idx}/{total}] Skipping {model_name} x {task_name} (seed={seed}) (already done)")
            continue
        
        print(f"\n[{idx}/{total}] Running {model_name} x {task_name}" + (f" (seed={seed})" if seed is not None else ""))
        try:
            result = run_single_experiment(model_name, task_name, device, train_config, seed=seed)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR: {model_name} x {task_name}: {e}")
            import traceback
            traceback.print_exc()

        # Save after each experiment
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Saved {len(all_results)}/{total} results")

    print(f"\n{'='*70}")
    print(f"ALL {total} EXPERIMENTS COMPLETE")
    print(f"{'='*70}")

    # Summary
    print("\n--- Task Coverage ---")
    for r in all_results:
        if r.get("seed") is None:
            print(f"  {r['model']:22s} x {r['task']:14s}: {r['best_val_seq_acc']*100:>7.2f}%  {r['wall_time_s']:.1f}s")

    print("\n--- Stability Test (bitwise_add) ---")
    for model_name in stability_models:
        accs = [r['best_val_seq_acc']*100 for r in all_results if r['model'] == model_name and r.get('seed') is not None]
        grads = [r['grad_norm_summary']['mean'] for r in all_results if r['model'] == model_name and r.get('seed') is not None]
        if accs:
            print(f"  {model_name:22s}: accs={[f'{a:.1f}%' for a in accs]}  grad_means={[f'{g:.1f}' for g in grads]}")
