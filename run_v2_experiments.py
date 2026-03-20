"""
Experiment v2: Test gated and adaptive Spinformer variants.

Hypothesis: Gated global rotation should fix the bitwise_add catastrophe
while preserving sorting gains. Adaptive per-token gating may further improve
by allowing different tokens to receive different rotation amounts.

Focus on sorting (discriminative) and bitwise_add (stability test).
Include baseline + best prior variants for comparison reference.
"""

import torch
import torch.nn as nn
import time
import json
import os
from collections import defaultdict

from ablation_suite.config import MODEL_CONFIGS, TASK_CONFIGS, TrainingConfig, ModelConfig, TaskConfig
from ablation_suite.train import create_model, Trainer, count_parameters, train_model


def run_single_experiment(model_name, task_name, device, train_config):
    """Run a single (model, task) experiment and return results dict."""
    model_cfg = MODEL_CONFIGS[model_name]
    task_cfg = TASK_CONFIGS[task_name]
    effective_config = train_config.for_task(task_name)

    print(f"\n{'#'*70}")
    print(f"# {model_name} × {task_name}")
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

    print(f"\n>>> {model_name} × {task_name}: "
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

    # Models to test: new variants + key references
    models = [
        "baseline",
        "spin_local_mlp",     # best safe variant from v1
        "spin_global_mlp",    # best overall from v1 (but unstable)
        "spin_gated_mlp",     # NEW: gated global rotation
        "spin_adaptive_mlp",  # NEW: adaptive per-token gating
    ]

    # Focus on discriminative tasks
    tasks = ["sorting", "bitwise_add"]

    results_file = "ablation_v2_results.json"
    all_results = []
    completed_keys = set()
    
    if os.path.exists(results_file):
        with open(results_file) as f:
            all_results = json.load(f)
        completed_keys = {(r["model"], r["task"]) for r in all_results}
        if completed_keys:
            print(f"Resuming: {len(completed_keys)} experiments already done")

    total = len(models) * len(tasks)
    idx = 0

    for task_name in tasks:
        for model_name in models:
            idx += 1
            if (model_name, task_name) in completed_keys:
                print(f"\n[{idx}/{total}] Skipping {model_name} × {task_name} (already done)")
                continue
            
            print(f"\n[{idx}/{total}] Running {model_name} × {task_name}...")
            try:
                result = run_single_experiment(model_name, task_name, device, train_config)
                all_results.append(result)
            except Exception as e:
                print(f"ERROR: {model_name} × {task_name}: {e}")
                import traceback
                traceback.print_exc()

            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*80}")
    print("V2 RESULTS SUMMARY")
    print(f"{'='*80}")
    
    for task_name in tasks:
        print(f"\n--- {task_name} ---")
        base = next((r for r in all_results if r['model'] == 'baseline' and r['task'] == task_name), None)
        for r in all_results:
            if r['task'] == task_name:
                acc = r['best_val_seq_acc'] * 100
                delta = f"({(r['best_val_seq_acc'] - base['best_val_seq_acc']) * 100:+.1f}pp)" if base and r['model'] != 'baseline' else ""
                grad = r.get('grad_norm_summary', {})
                grad_str = f"grad_mean={grad.get('mean', 0):.2f}" if grad else ""
                gate_info = ""
                if r.get('spin_stats'):
                    gates = []
                    for layer, stats in r['spin_stats'].items():
                        for k, v in stats.items():
                            if 'gate' in k and isinstance(v, float):
                                gates.append(f"{v:.3f}")
                    if gates:
                        gate_info = f"gates=[{','.join(gates)}]"
                print(f"  {r['model']:22s}: {acc:>7.2f}% {delta:>10s}  {r['wall_time_s']:>7.1f}s  {grad_str}  {gate_info}")
