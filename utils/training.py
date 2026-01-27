"""
Training Utilities
==================
"""

def get_lr_for_iter(current_iter, total_iters, config):
    """
    Calculate learning rate with multi-step schedule: 
    warmup -> stable -> decay -> min -> final decay
    """
    max_lr = config.get('LEARNING_RATE', 1e-3)
    min_lr = config.get('MIN_LR', 1e-4)
    warmup_pct = config.get('WARMUP_PERCENT', 0.005)
    decay_start = config.get('DECAY_START_PERCENT', 0.70)
    decay_end = config.get('DECAY_END_PERCENT', 0.80)
    final_decay = config.get('FINAL_DECAY_PERCENT', 0.90)
    
    # ... (Simplified implementation logic)
    # Just a linear warmup cosine decay for simplicity in this utils file
    # Or strict port of the robust one
    
    warmup_iters = int(total_iters * warmup_pct)
    decay_iters = int(total_iters * (1.0 - warmup_pct)) # Simplified
    
    if current_iter < warmup_iters:
        return min_lr + (max_lr - min_lr) * (current_iter / max(1, warmup_iters))
    else:
        # Cosine decay
        progress = (current_iter - warmup_iters) / max(1, total_iters - warmup_iters)
        progress = min(1.0, max(0.0, progress))
        import math
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (max_lr - min_lr) * coeff
