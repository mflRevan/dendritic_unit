#!/usr/bin/env python3
"""
GeoField Language Modeling Experiments
======================================
Char-level and GPT-2-tokenized WikiText-2 ablations.

Replace mode: weight matrices are fully generated from rotated/scaled 3D
latent coordinates. No residual — pure geometric weight generation.

Targets: All subsets of {Q, K, V, O} attention projections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import json
import time
import math
import os
import sys
import random
import numpy as np
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.geofield_transformer import GeoFieldTransformer
from model.transformer import Transformer
from utils.data_utils import get_wikitext_char_dataloader, get_wikitext_dataloader

# ========================================================================
#  Global config
# ========================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architecture
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
SEQ_LEN = 256
GEO_NUM_COORDS = 32

# Training
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 200

RESULTS_DIR = "lm_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ========================================================================
#  Model definitions
# ========================================================================

def make_model(cfg, vocab_size):
    """Create model from config dict."""
    d = cfg.get("d_model", D_MODEL)
    h = cfg.get("n_heads", N_HEADS)
    n = cfg.get("n_layers", N_LAYERS)

    if cfg.get("arch", "transformer") == "geofield":
        return GeoFieldTransformer(
            vocab_size=vocab_size,
            seq_length=SEQ_LEN * 2,
            dim=d,
            num_heads=h,
            num_layers=n,
            geo_target=cfg["geo_target"],
            geo_mode=cfg.get("geo_mode", "replace"),
            geo_conditioning=cfg.get("geo_conditioning", "static"),
            geo_num_coords=cfg.get("geo_num_coords", GEO_NUM_COORDS),
            geo_use_scale=cfg.get("geo_use_scale", True),
            geo_use_pivot_offset=cfg.get("geo_use_pivot_offset", False),
            geo_cond_source=cfg.get("geo_cond_source", "mean_pool"),
            geo_shared_controller=cfg.get("geo_shared_controller", False),
            geo_controller_type=cfg.get("geo_controller_type", "local"),
            geo_rotation=cfg.get("geo_rotation", "quaternion"),
            geo_coord_dim=cfg.get("geo_coord_dim", 3),
            geo_rank=cfg.get("geo_rank", 0),
            geo_conditioned_layers=cfg.get("geo_conditioned_layers", "all"),
        )
    else:
        return Transformer(
            vocab_size=vocab_size,
            seq_length=SEQ_LEN * 2,
            dim=d,
            num_heads=h,
            num_layers=n,
        )


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ========================================================================
#  Training loop
# ========================================================================

def train_epoch(model, loader, optimizer, scheduler, scaler, warmup_steps, global_step):
    model.train()
    total_loss = 0.0
    n_tokens = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP).item()
        scaler.step(optimizer)
        scaler.update()

        # Warmup: linear ramp then cosine
        global_step += 1
        if global_step <= warmup_steps:
            lr_scale = global_step / warmup_steps
            for pg in optimizer.param_groups:
                pg['lr'] = LR * lr_scale
        else:
            scheduler.step()

        total_loss += loss.item() * targets.numel()
        n_tokens += targets.numel()

    return total_loss / n_tokens, global_step


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    n_tokens = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction='sum',
            )

        total_loss += loss.item()
        n_tokens += targets.numel()

    avg_loss = total_loss / n_tokens
    ppl = math.exp(min(avg_loss, 20))  # cap to prevent overflow
    bpc = avg_loss / math.log(2)
    return avg_loss, ppl, bpc


def collect_geo_stats(model):
    """Safely get geo diagnostics from a (possibly compiled) model."""
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    if hasattr(m, 'get_geo_stats'):
        return m.get_geo_stats()
    return {}


def collect_grad_stats(model):
    """Compute gradient norm statistics."""
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            norms.append(p.grad.data.norm(2).item())
    if not norms:
        return {}
    return {
        "grad_norm_mean": np.mean(norms),
        "grad_norm_max": max(norms),
        "grad_norm_median": float(np.median(norms)),
    }


# ========================================================================
#  Experiment runner
# ========================================================================

def run_experiment(name, cfg, train_loader, val_loader, vocab_size, seed=42):
    """Train one model and return full history."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    set_seed(seed)

    model = make_model(cfg, vocab_size).to(DEVICE)
    n_params = count_params(model)
    print(f"  Parameters: {n_params:,}")
    print(f"  Config: {cfg}")

    # Compile
    compiled = torch.compile(model, mode="default")

    optimizer = AdamW(
        compiled.parameters(), lr=LR,
        weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95),
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(total_steps - WARMUP_STEPS, 1),
        eta_min=LR * 0.01,
    )
    scaler = torch.amp.GradScaler('cuda')

    history = {
        "name": name,
        "config": cfg,
        "params": n_params,
        "epochs": [],
    }

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(EPOCHS):
        t0 = time.time()

        train_loss, global_step = train_epoch(
            compiled, train_loader, optimizer, scheduler, scaler,
            WARMUP_STEPS, global_step,
        )
        val_loss, val_ppl, val_bpc = evaluate(compiled, val_loader)

        # Collect diagnostics
        geo_stats = collect_geo_stats(compiled)
        grad_stats = collect_grad_stats(compiled)

        elapsed = time.time() - t0

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_ppl": round(val_ppl, 2),
            "val_bpc": round(val_bpc, 4),
            "time": round(elapsed, 1),
            "lr": round(optimizer.param_groups[0]['lr'], 8),
        }
        if geo_stats:
            epoch_data["geo_stats"] = geo_stats
        if grad_stats:
            epoch_data["grad_stats"] = {
                k: round(v, 4) for k, v in grad_stats.items()
            }

        history["epochs"].append(epoch_data)

        marker = " *" if val_loss <= best_val_loss else ""
        print(
            f"  Epoch {epoch+1:2d}/{EPOCHS} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"PPL: {val_ppl:8.2f} | "
            f"BPC: {val_bpc:.4f} | "
            f"Time: {elapsed:.1f}s{marker}"
        )

    history["best_val_loss"] = round(best_val_loss, 6)
    history["best_val_ppl"] = round(math.exp(min(best_val_loss, 20)), 2)
    history["best_val_bpc"] = round(best_val_loss / math.log(2), 4)
    history["total_time"] = round(sum(e["time"] for e in history["epochs"]), 1)

    print(f"\n  Best: PPL={history['best_val_ppl']:.2f}  BPC={history['best_val_bpc']:.4f}")

    # Cleanup
    del compiled, model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()

    return history


# ========================================================================
#  Phase definitions
# ========================================================================

# Phase 1: single projections (char-level)
PHASE1_CHAR = OrderedDict([
    ("baseline",        {"arch": "transformer"}),
    ("geo_q",           {"arch": "geofield", "geo_target": "q"}),
    ("geo_k",           {"arch": "geofield", "geo_target": "k"}),
    ("geo_v",           {"arch": "geofield", "geo_target": "v"}),
    ("geo_o",           {"arch": "geofield", "geo_target": "o"}),
    ("geo_o_noscale",   {"arch": "geofield", "geo_target": "o", "geo_use_scale": False}),
    ("geo_o_cond",      {"arch": "geofield", "geo_target": "o", "geo_conditioning": "seq_conditioned"}),
])

# Phase 2: conditioned singles + pairs + big baseline (char-level)
PHASE2_CHAR = OrderedDict([
    # Conditioned single projections (Phase 1 showed conditioning dominates)
    ("geo_q_cond",    {"arch": "geofield", "geo_target": "q", "geo_conditioning": "seq_conditioned"}),
    ("geo_k_cond",    {"arch": "geofield", "geo_target": "k", "geo_conditioning": "seq_conditioned"}),
    ("geo_v_cond",    {"arch": "geofield", "geo_target": "v", "geo_conditioning": "seq_conditioned"}),
    # Conditioned pairs
    ("geo_vo_cond",   {"arch": "geofield", "geo_target": "vo", "geo_conditioning": "seq_conditioned"}),
    ("geo_qk_cond",   {"arch": "geofield", "geo_target": "qk", "geo_conditioning": "seq_conditioned"}),
    ("geo_qo_cond",   {"arch": "geofield", "geo_target": "qo", "geo_conditioning": "seq_conditioned"}),
    # Conditioned full
    ("geo_qkvo_cond", {"arch": "geofield", "geo_target": "qkvo", "geo_conditioning": "seq_conditioned"}),
    # Parameter-matched baselines (control for param count ~7.4M)
    ("big_baseline_w",  {"arch": "transformer", "d_model": 256, "n_heads": 8, "n_layers": 4}),
    ("big_baseline_d",  {"arch": "transformer", "d_model": 128, "n_heads": 4, "n_layers": 24}),
])

# Phase 3: static pairs for comparison + extended conditioned (char-level)
PHASE3_CHAR = OrderedDict([
    # Static pairs (to compare conditioning vs static for pairs)
    ("geo_vo",   {"arch": "geofield", "geo_target": "vo"}),
    ("geo_qk",   {"arch": "geofield", "geo_target": "qk"}),
    ("geo_qo",   {"arch": "geofield", "geo_target": "qo"}),
    ("geo_qkvo", {"arch": "geofield", "geo_target": "qkvo"}),
])

# Phase 4: Field size ablation on GPT-2 (Finding 43: data-aware field sizing)
# Test if smaller geo_num_coords fixes the parameter explosion on GPT-2
PHASE4_GPT2 = OrderedDict([
    # VO conditioned with varying field sizes
    ("vo_cond_c4",    {"arch": "geofield", "geo_target": "vo", "geo_conditioning": "seq_conditioned", "geo_num_coords": 4}),
    ("vo_cond_c8",    {"arch": "geofield", "geo_target": "vo", "geo_conditioning": "seq_conditioned", "geo_num_coords": 8}),
    ("vo_cond_c16",   {"arch": "geofield", "geo_target": "vo", "geo_conditioning": "seq_conditioned", "geo_num_coords": 16}),
    # QKVO conditioned with smaller fields — can we recover the char-level advantage?
    ("qkvo_cond_c4",  {"arch": "geofield", "geo_target": "qkvo", "geo_conditioning": "seq_conditioned", "geo_num_coords": 4}),
    ("qkvo_cond_c8",  {"arch": "geofield", "geo_target": "qkvo", "geo_conditioning": "seq_conditioned", "geo_num_coords": 8}),
    ("qkvo_cond_c16", {"arch": "geofield", "geo_target": "qkvo", "geo_conditioning": "seq_conditioned", "geo_num_coords": 16}),
])

# Phase 5: Field size ablation on char-level (control: does field size matter when data is plentiful?)
PHASE5_CHAR = OrderedDict([
    ("vo_cond_c4",    {"arch": "geofield", "geo_target": "vo", "geo_conditioning": "seq_conditioned", "geo_num_coords": 4}),
    ("vo_cond_c8",    {"arch": "geofield", "geo_target": "vo", "geo_conditioning": "seq_conditioned", "geo_num_coords": 8}),
    ("vo_cond_c16",   {"arch": "geofield", "geo_target": "vo", "geo_conditioning": "seq_conditioned", "geo_num_coords": 16}),
    ("qkvo_cond_c4",  {"arch": "geofield", "geo_target": "qkvo", "geo_conditioning": "seq_conditioned", "geo_num_coords": 4}),
    ("qkvo_cond_c8",  {"arch": "geofield", "geo_target": "qkvo", "geo_conditioning": "seq_conditioned", "geo_num_coords": 8}),
    ("qkvo_cond_c16", {"arch": "geofield", "geo_target": "qkvo", "geo_conditioning": "seq_conditioned", "geo_num_coords": 16}),
])

# ========================================================================
#  Tier 2: Conditioning source ablation (char-level)
# ========================================================================

# Phase 6: Conditioning source ablation
_COND_BASE = {"arch": "geofield", "geo_target": "qkvo", "geo_conditioning": "seq_conditioned"}

PHASE6_COND_SOURCE = OrderedDict([
    ("src_mean_pool",     {**_COND_BASE, "geo_cond_source": "mean_pool"}),
    ("src_last_token",    {**_COND_BASE, "geo_cond_source": "last_token"}),
    ("src_first_token",   {**_COND_BASE, "geo_cond_source": "first_token"}),
    ("src_max_pool",      {**_COND_BASE, "geo_cond_source": "max_pool"}),
    ("src_attn_pool",     {**_COND_BASE, "geo_cond_source": "attn_pool"}),
    ("src_detached_mean", {**_COND_BASE, "geo_cond_source": "detached_mean"}),
    ("src_per_token",     {**_COND_BASE, "geo_cond_source": "per_token"}),
])

# Phase 7: Controller design ablation
PHASE7_CONTROLLER = OrderedDict([
    ("ctrl_separate",    {**_COND_BASE}),
    ("ctrl_shared",      {**_COND_BASE, "geo_shared_controller": True}),
    ("mode_replace",     {**_COND_BASE, "geo_mode": "replace"}),
    ("mode_residual",    {**_COND_BASE, "geo_mode": "residual"}),
    ("o_cond_separate",  {"arch": "geofield", "geo_target": "o", "geo_conditioning": "seq_conditioned"}),
    ("o_cond_residual",  {"arch": "geofield", "geo_target": "o", "geo_conditioning": "seq_conditioned", "geo_mode": "residual"}),
])

# Phase 8: Cross-layer controller ablation (Tier 3)
PHASE8_CONTROLLER_TYPE = OrderedDict([
    ("ctrl_local",      {**_COND_BASE, "geo_controller_type": "local"}),
    ("ctrl_first_only", {**_COND_BASE, "geo_controller_type": "first_only"}),
    ("ctrl_ema",        {**_COND_BASE, "geo_controller_type": "ema"}),
    ("ctrl_gru",        {**_COND_BASE, "geo_controller_type": "gru"}),
])

# Phase 9: Rotation type & coordinate dimension ablation (Tier 4)
PHASE9_ROTATION = OrderedDict([
    ("rot_quat_d3",     {**_COND_BASE, "geo_rotation": "quaternion", "geo_coord_dim": 3}),
    ("rot_cayley_d3",   {**_COND_BASE, "geo_rotation": "cayley", "geo_coord_dim": 3}),
    ("rot_cayley_d4",   {**_COND_BASE, "geo_rotation": "cayley", "geo_coord_dim": 4}),
    ("rot_cayley_d6",   {**_COND_BASE, "geo_rotation": "cayley", "geo_coord_dim": 6}),
    ("rot_linear_d3",   {**_COND_BASE, "geo_rotation": "linear", "geo_coord_dim": 3}),
])

# Phase 10: Decoder rank ablation (Tier 4)
PHASE10_RANK = OrderedDict([
    ("rank_full",   {**_COND_BASE, "geo_rank": 0}),
    ("rank_4",      {**_COND_BASE, "geo_rank": 4}),
    ("rank_8",      {**_COND_BASE, "geo_rank": 8}),
    ("rank_16",     {**_COND_BASE, "geo_rank": 16}),
])

# Phase 11: Layer-role study (which layers benefit from conditioning)
# 4-layer model: test conditioning at different layer subsets
PHASE11_LAYER_ROLE = OrderedDict([
    ("layers_all",       {**_COND_BASE, "geo_conditioned_layers": "all"}),       # all conditioned (reference)
    ("layers_lower",     {**_COND_BASE, "geo_conditioned_layers": "0,1"}),       # only lower layers
    ("layers_upper",     {**_COND_BASE, "geo_conditioned_layers": "2,3"}),       # only upper layers
    ("layers_first",     {**_COND_BASE, "geo_conditioned_layers": "0"}),         # only first layer
    ("layers_last",      {**_COND_BASE, "geo_conditioned_layers": "3"}),         # only last layer
    ("layers_middle",    {**_COND_BASE, "geo_conditioned_layers": "1,2"}),       # only middle layers
    ("layers_alternating", {**_COND_BASE, "geo_conditioned_layers": "0,2"}),     # alternating (even)
])

# Phase 12: Cache-safe target study (which projections need conditioning)
# Key question: can we keep K/V static (cacheable) while only conditioning Q/O?
PHASE12_CACHE_SAFE = OrderedDict([
    ("cache_qkvo_cond",  {**_COND_BASE}),                                                                        # all conditioned (reference)
    ("cache_o_only",     {"arch": "geofield", "geo_target": "o", "geo_conditioning": "seq_conditioned"}),         # only O conditioned (full KV cache)
    ("cache_qo_cond",    {"arch": "geofield", "geo_target": "qo", "geo_conditioning": "seq_conditioned"}),       # Q,O conditioned (K,V static → full KV cache)
    ("cache_vo_cond",    {"arch": "geofield", "geo_target": "vo", "geo_conditioning": "seq_conditioned"}),        # V,O conditioned (K static)
    ("cache_qvo_cond",   {"arch": "geofield", "geo_target": "qvo", "geo_conditioning": "seq_conditioned"}),      # Q,V,O conditioned (K static)
])

# Phase 13: Combined best configuration
# Take winners from each ablation dimension and test combinations
PHASE13_COMBINED = OrderedDict([
    ("best_base",         {**_COND_BASE}),                                                                        # reference: qkvo, conditioned, mean_pool, local, quat_d3, full rank
    ("best_cayley",       {**_COND_BASE, "geo_rotation": "cayley"}),                                              # swap in cayley (≈quat)
    ("best_ema",          {**_COND_BASE, "geo_controller_type": "ema"}),                                          # swap in EMA controller
    ("best_cayley_ema",   {**_COND_BASE, "geo_rotation": "cayley", "geo_controller_type": "ema"}),                # cayley + EMA
    ("best_partial",      {**_COND_BASE, "geo_conditioned_layers": "0,1"}),                                       # only lower layers (test after Phase 11)
])


def load_results(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_results(results, path):
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def print_summary(results, title="RESULTS"):
    """Print a formatted summary table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    header = f"{'Model':<20} {'Params':>10} {'Best PPL':>10} {'Best BPC':>10} {'Final PPL':>10} {'Time':>8}"
    print(header)
    print("-" * 80)

    # Sort by best BPC
    items = sorted(results.items(), key=lambda x: x[1].get("best_val_bpc", 999))
    for name, r in items:
        final_ppl = r["epochs"][-1]["val_ppl"] if r.get("epochs") else 0
        print(
            f"{name:<20} "
            f"{r.get('params', 0):>10,} "
            f"{r.get('best_val_ppl', 0):>10.2f} "
            f"{r.get('best_val_bpc', 0):>10.4f} "
            f"{final_ppl:>10.2f} "
            f"{r.get('total_time', 0):>7.0f}s"
        )
    print()


def run_phase(phase_name, models, data_loaders, results_path, seed=42):
    """Run a phase of experiments, resuming from saved results."""
    train_loader, val_loader, vocab_size = data_loaders
    results = load_results(results_path)

    for name, cfg in models.items():
        if name in results:
            print(f"  [{phase_name}] Skipping {name} (already done)")
            continue

        history = run_experiment(name, cfg, train_loader, val_loader, vocab_size, seed)
        results[name] = history
        save_results(results, results_path)

    print_summary(results, f"{phase_name} SUMMARY")
    return results


# ========================================================================
#  Main
# ========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GeoField LM Experiments")
    parser.add_argument("--phase", type=str, default="1",
                        help="Phase to run: 1, 2, 3, 4, 5, all, or gpt2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    if args.epochs:
        global EPOCHS
        EPOCHS = args.epochs
    if args.batch_size:
        global BATCH_SIZE
        BATCH_SIZE = args.batch_size

    phases = args.phase.split(",")

    # ---- Char-level phases ----
    if any(p in phases for p in ["1", "2", "3", "5", "6", "7", "8", "9", "10", "11", "12", "13", "all"]):
        print("=" * 70)
        print("  Loading char-level WikiText-2...")
        print("=" * 70)
        train_loader, val_loader, vocab_size, itos = get_wikitext_char_dataloader(
            seq_length=SEQ_LEN, batch_size=BATCH_SIZE,
        )
        char_data = (train_loader, val_loader, vocab_size)

        if "1" in phases or "all" in phases:
            run_phase("PHASE 1 (Char, Singles)",
                      PHASE1_CHAR, char_data,
                      os.path.join(RESULTS_DIR, "phase1_char.json"),
                      seed=args.seed)

        if "2" in phases or "all" in phases:
            run_phase("PHASE 2 (Char, Pairs)",
                      PHASE2_CHAR, char_data,
                      os.path.join(RESULTS_DIR, "phase2_char.json"),
                      seed=args.seed)

        if "3" in phases or "all" in phases:
            run_phase("PHASE 3 (Char, Triples+Full)",
                      PHASE3_CHAR, char_data,
                      os.path.join(RESULTS_DIR, "phase3_char.json"),
                      seed=args.seed)

        if "5" in phases:
            run_phase("PHASE 5 (Char, Field Size Ablation)",
                      PHASE5_CHAR, char_data,
                      os.path.join(RESULTS_DIR, "phase5_char.json"),
                      seed=args.seed)

        if "6" in phases:
            run_phase("PHASE 6 (Char, Conditioning Source Ablation)",
                      PHASE6_COND_SOURCE, char_data,
                      os.path.join(RESULTS_DIR, "phase6_cond_source.json"),
                      seed=args.seed)

        if "7" in phases:
            run_phase("PHASE 7 (Char, Controller Design Ablation)",
                      PHASE7_CONTROLLER, char_data,
                      os.path.join(RESULTS_DIR, "phase7_controller.json"),
                      seed=args.seed)

        if "8" in phases:
            run_phase("PHASE 8 (Char, Cross-Layer Controllers)",
                      PHASE8_CONTROLLER_TYPE, char_data,
                      os.path.join(RESULTS_DIR, "phase8_ctrl_type.json"),
                      seed=args.seed)

        if "9" in phases:
            run_phase("PHASE 9 (Char, Rotation Type & Coord Dim)",
                      PHASE9_ROTATION, char_data,
                      os.path.join(RESULTS_DIR, "phase9_rotation.json"),
                      seed=args.seed)

        if "10" in phases:
            run_phase("PHASE 10 (Char, Decoder Rank Ablation)",
                      PHASE10_RANK, char_data,
                      os.path.join(RESULTS_DIR, "phase10_rank.json"),
                      seed=args.seed)

        if "11" in phases:
            run_phase("PHASE 11 (Char, Layer-Role Study)",
                      PHASE11_LAYER_ROLE, char_data,
                      os.path.join(RESULTS_DIR, "phase11_layer_role.json"),
                      seed=args.seed)

        if "12" in phases:
            run_phase("PHASE 12 (Char, Cache-Safe Targets)",
                      PHASE12_CACHE_SAFE, char_data,
                      os.path.join(RESULTS_DIR, "phase12_cache_safe.json"),
                      seed=args.seed)

        if "13" in phases:
            run_phase("PHASE 13 (Char, Combined Best)",
                      PHASE13_COMBINED, char_data,
                      os.path.join(RESULTS_DIR, "phase13_combined.json"),
                      seed=args.seed)

    # ---- GPT-2 tokenizer phase ----
    if any(p in phases for p in ["gpt2", "4", "all"]):
        print("=" * 70)
        print("  Loading GPT-2-tokenized WikiText-2...")
        print("=" * 70)
        train_loader, val_loader, gpt2_vocab = get_wikitext_dataloader(
            seq_length=SEQ_LEN, batch_size=BATCH_SIZE,
        )
        gpt2_data = (train_loader, val_loader, gpt2_vocab)

        if "gpt2" in phases or "all" in phases:
            GPT2_MODELS = OrderedDict([
                ("baseline",        {"arch": "transformer"}),
                ("geo_o",           {"arch": "geofield", "geo_target": "o"}),
                ("geo_o_cond",      {"arch": "geofield", "geo_target": "o", "geo_conditioning": "seq_conditioned"}),
                ("geo_vo_cond",     {"arch": "geofield", "geo_target": "vo", "geo_conditioning": "seq_conditioned"}),
                ("geo_qkvo_cond",   {"arch": "geofield", "geo_target": "qkvo", "geo_conditioning": "seq_conditioned"}),
                ("big_baseline_d",  {"arch": "transformer", "d_model": 128, "n_heads": 4, "n_layers": 24}),
            ])

            run_phase("GPT-2 TOKENIZER",
                      GPT2_MODELS, gpt2_data,
                      os.path.join(RESULTS_DIR, "gpt2_results.json"),
                      seed=args.seed)

        if "4" in phases:
            run_phase("PHASE 4 (GPT-2, Field Size Ablation)",
                      PHASE4_GPT2, gpt2_data,
                      os.path.join(RESULTS_DIR, "phase4_gpt2.json"),
                      seed=args.seed)


if __name__ == "__main__":
    main()
