"""
Full ablation experiment runner: 5 models × 5 tasks.
Collects results, timing, gradient stats, and spin dynamics.
"""

import torch
import torch.nn as nn
import time
import json
import sys
from collections import defaultdict

from ablation_suite.config import MODEL_CONFIGS, TASK_CONFIGS, TrainingConfig, ModelConfig, TaskConfig
from ablation_suite.train import create_model, Trainer, count_parameters, train_model


def run_single_experiment(model_name, task_name, device, train_config):
    """Run a single (model, task) experiment and return results dict."""
    model_cfg = MODEL_CONFIGS[model_name]
    task_cfg = TASK_CONFIGS[task_name]
    
    # Apply per-task training overrides
    effective_config = train_config.for_task(task_name)

    print(f"\n{'#'*70}")
    print(f"# {model_name} × {task_name}")
    print(f"{'#'*70}")

    start = time.time()
    model, metrics, task = train_model(model_cfg, task_cfg, effective_config, device)
    wall_time = time.time() - start

    n_params = count_parameters(model)

    # Gather spin stats if spinformer
    spin_stats = {}
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    if hasattr(raw_model, 'get_spin_stats'):
        spin_stats = raw_model.get_spin_stats()

    # Gradient norms per layer
    grad_norms = {}
    raw_model.eval()
    raw_model.train()
    # Re-run a single forward+backward to get gradient norms
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
        "final_val_token_acc": metrics.val_token_accuracies[-1] if metrics.val_token_accuracies else 0.0,
        "final_train_loss": metrics.train_losses[-1] if metrics.train_losses else float("inf"),
        "final_val_loss": metrics.val_losses[-1] if metrics.val_losses else float("inf"),
        "val_seq_accs": metrics.val_accuracies,
        "val_token_accs": metrics.val_token_accuracies,
        "train_losses": metrics.train_losses,
        "val_losses": metrics.val_losses,
        "epoch_times": metrics.epoch_times,
        "spin_stats": {k: {sk: sv for sk, sv in v.items()} for k, v in spin_stats.items()} if spin_stats else {},
        "grad_norm_summary": {
            "mean": sum(grad_norms.values()) / len(grad_norms) if grad_norms else 0,
            "max": max(grad_norms.values()) if grad_norms else 0,
            "min": min(grad_norms.values()) if grad_norms else 0,
        },
    }

    # Print summary
    print(f"\n>>> {model_name} × {task_name}: "
          f"Best Seq Acc = {result['best_val_seq_acc']*100:.2f}%, "
          f"Token Acc = {result['best_val_token_acc']*100:.2f}%, "
          f"Time = {wall_time:.1f}s")

    return result


def print_results_table(results):
    """Print a summary table of all results."""
    # Group by task
    tasks = list(TASK_CONFIGS.keys())
    models = list(MODEL_CONFIGS.keys())

    print(f"\n{'='*100}")
    print("RESULTS SUMMARY: Best Validation Sequence Accuracy (%)")
    print(f"{'='*100}")
    header = f"{'Model':<22s}" + "".join(f"{t:>15s}" for t in tasks) + f"{'Avg':>10s}"
    print(header)
    print("-" * len(header))

    for m in models:
        row = f"{m:<22s}"
        accs = []
        for t in tasks:
            key = f"{m}_{t}"
            r = next((x for x in results if x["model"] == m and x["task"] == t), None)
            if r:
                acc = r["best_val_seq_acc"] * 100
                accs.append(acc)
                row += f"{acc:>15.2f}"
            else:
                row += f"{'N/A':>15s}"
        if accs:
            row += f"{sum(accs)/len(accs):>10.2f}"
        print(row)

    print(f"{'='*100}")

    # Timing
    print(f"\nTIMING (seconds per full training run):")
    print(f"{'Model':<22s}" + "".join(f"{t:>15s}" for t in tasks))
    print("-" * 100)
    for m in models:
        row = f"{m:<22s}"
        for t in tasks:
            r = next((x for x in results if x["model"] == m and x["task"] == t), None)
            if r:
                row += f"{r['wall_time_s']:>15.1f}"
            else:
                row += f"{'N/A':>15s}"
        print(row)


def format_results_markdown(results):
    """Format results as markdown table for RESEARCH.md."""
    tasks = list(TASK_CONFIGS.keys())
    models = list(MODEL_CONFIGS.keys())

    lines = []
    lines.append("| Model | " + " | ".join(tasks) + " | Avg |")
    lines.append("|" + "---|" * (len(tasks) + 2))

    for m in models:
        accs = []
        cells = []
        for t in tasks:
            r = next((x for x in results if x["model"] == m and x["task"] == t), None)
            if r:
                acc = r["best_val_seq_acc"] * 100
                accs.append(acc)
                cells.append(f"{acc:.1f}%")
            else:
                cells.append("N/A")
        avg = f"{sum(accs)/len(accs):.1f}%" if accs else "N/A"
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {avg} |")

    # Token accuracy table
    lines.append("")
    lines.append("**Token Accuracy (best)**")
    lines.append("")
    lines.append("| Model | " + " | ".join(tasks) + " | Avg |")
    lines.append("|" + "---|" * (len(tasks) + 2))

    for m in models:
        accs = []
        cells = []
        for t in tasks:
            r = next((x for x in results if x["model"] == m and x["task"] == t), None)
            if r:
                acc = r["best_val_token_acc"] * 100
                accs.append(acc)
                cells.append(f"{acc:.1f}%")
            else:
                cells.append("N/A")
        avg = f"{sum(accs)/len(accs):.1f}%" if accs else "N/A"
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {avg} |")

    # Timing table
    lines.append("")
    lines.append("**Wall-clock time (seconds)**")
    lines.append("")
    lines.append("| Model | " + " | ".join(tasks) + " |")
    lines.append("|" + "---|" * (len(tasks) + 1))

    for m in models:
        cells = []
        for t in tasks:
            r = next((x for x in results if x["model"] == m and x["task"] == t), None)
            if r:
                cells.append(f"{r['wall_time_s']:.1f}s")
            else:
                cells.append("N/A")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_config = TrainingConfig(
        num_epochs=10,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=0.1,
        grad_clip=1.0,
        use_compile=True,
        task_overrides={
            # Parity can be harder, give it more time
            "parity": {"num_epochs": 20},
        },
    )

    all_results = []
    tasks = list(TASK_CONFIGS.keys())
    models = list(MODEL_CONFIGS.keys())
    
    # Resume from previous run if exists
    results_file = "ablation_results.json"
    completed_keys = set()
    try:
        import os
        if os.path.exists(results_file):
            with open(results_file) as f:
                all_results = json.load(f)
            completed_keys = {(r["model"], r["task"]) for r in all_results}
            if completed_keys:
                print(f"Resuming: {len(completed_keys)} experiments already done")
    except (json.JSONDecodeError, KeyError):
        all_results = []
        completed_keys = set()

    total = len(models) * len(tasks)
    idx = 0

    for task_name in tasks:
        for model_name in models:
            idx += 1
            
            # Skip already completed experiments
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

            # Save intermediate results
            with open("ablation_results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    # Print final summary
    print_results_table(all_results)

    # Save markdown
    md = format_results_markdown(all_results)
    print(f"\n\nMARKDOWN TABLE:\n{md}")

    # Save to file
    with open("ablation_results_table.md", "w") as f:
        f.write(md)

    print(f"\nResults saved to ablation_results.json and ablation_results_table.md")
