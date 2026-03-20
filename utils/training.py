"""
Training Utilities
==================
"""

def get_lr_for_iter(current_iter, total_iters, config):
    """
    Calculate learning rate with WSD schedule:
    Warmup (linear) -> Stable (constant) -> Decay (linear)
    """
    max_lr = config.get('LEARNING_RATE', 3e-4)
    min_lr = config.get('MIN_LR', 0.0)
    
    # Percentages
    warmup_pct = config.get('WARMUP_PERCENT', 0.1)
    stable_pct = config.get('STABLE_PERCENT', 0.8)
    # Decay percent is the remainder
    
    warmup_iters = int(total_iters * warmup_pct)
    decay_start_iter = int(total_iters * (warmup_pct + stable_pct))
    
    # 1. Linear Warmup
    if current_iter < warmup_iters:
        return (current_iter / max(1, warmup_iters)) * max_lr
        
    # 2. Stable Phase
    elif current_iter < decay_start_iter:
        return max_lr
        
    # 3. Linear Decay to Zero (or min_lr)
    else:
        decay_steps = total_iters - decay_start_iter
        progress = (current_iter - decay_start_iter) / max(1, decay_steps)
        progress = min(1.0, max(0.0, progress))
        # Linear decay: max_lr -> min_lr
        return max_lr - (progress * (max_lr - min_lr))

