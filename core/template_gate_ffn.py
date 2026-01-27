"""
Dendritic FFN Triton Kernel - Template-Gated Direct Projection.

This module provides a forward+backward implementation for dendritic FFN that computes:
    out = gate(x, template) * (x @ weights.T)

WITHOUT a separate parallel linear projection - the template-based gating IS the
full forward mixing projection. This differs from the fused_up kernel which computes:
    out = gate(x, template) * (x @ weights.T)
where gate modulates a separate linear projection.

In dendritic FFN mode:
- The gate mechanism itself provides the feature transformation
- Each output dimension h has learned templates that pattern-match against input windows
- The output is the gated sum of template dot products, directly used as features

Formula:
    dot[h, w] = x_window[w] · template[h, w, :]  # raw dot products ARE the scores
    prob[h, w] = softmax(|dot| / tau)
    out[h] = sum_w(prob[h, w] * dot[h, w]) * (1 + ln(1 + |mean(dot)|))

This is conceptually different from GLU-style gating:
- GLU: gate * data_projection (two parallel paths)
- Dendritic FFN: the gating mechanism IS the projection (single path)

Key properties:
1. No separate up projection weights - templates serve as both gates AND projections
2. Output is the weighted sum of template matches
3. Softmax competition determines which template(s) contribute most
4. Bipolar outputs: both positive and negative dot products contribute

Performance characteristics:
- Fewer parameters than GLU (no separate linear weights)
- Single kernel pass (no fusion needed, it's already minimal)
- Lower memory bandwidth (no weight matrix for linear projection)

Numerical Stability Optimizations:
- Uses bfloat16 (bf16) for tensor core matmuls instead of fp16 for better numerical
  stability with large score ranges during training
- Uses exp2 (base-2 exponentials) instead of exp for softmax calculations:
  exp(x) = 2^(x * log2(e)), which is hardware-optimized and more stable
- log2 used for coactivation term for consistency: ln(x) = log2(x) / log2(e)
- These changes provide improved stability for dynamic score ranges during training
"""

import torch
import triton
import triton.language as tl
from torch.autograd import Function


def _get_ffn_fwd_block_sizes(N, H, W, num_windows):
    """
    Select optimal block sizes for FFN forward kernel.
    
    IMPORTANT: FFN has NO linear projection, so register pressure is much lower
    than fused_up. This allows for LARGER block_h and potentially different N tuning.
    
    Empirically tuned results (RTX 5080, D=512, H=2048, W=64):
    - All batch sizes: block_h=32-64 is optimal (more ILP, less register pressure)
    - N=256:  (64, 32) best
    - N=512:  (64, 32) best
    - N=1024: (64, 32) best
    - N=2048: (64, 32) best
    - N=4096: (64, 32) best
    - N=8192: (64, 32) best
    
    Key insight: FFN benefits from larger block_h because it doesn't have
    the dot product accumulators of the linear projection.
    """
    # All sizes: use 64x32 as the sweet spot
    # Lower register pressure allows higher block_h
    block_n, block_h = 64, 32
    
    # Very small batches: reduce block_n but keep block_h large
    if N <= 32:
        block_n = 16
    elif N <= 64:
        block_n = 32
    
    # Many windows: reduce to avoid register spills from online softmax
    if num_windows > 8:
        block_n = min(block_n, 32)
        block_h = min(block_h, 32)
    
    return block_n, block_h


def _get_ffn_bwd_block_sizes(N, H, W, num_windows):
    """
    Select optimal block sizes for FFN backward kernel.
    
    IMPORTANT: FFN backward is lighter than fused_up backward because:
    1. No weight gradient computation (no tl.dot on [BLOCK_N, D] × [D, H])
    2. No atomic accumulation into weight gradient matrix
    3. Only x and template gradients via atomics
    
    Empirically tuned results (RTX 5080, D=512, H=2048, W=64):
    - N=256:  (16, 16) best → less atomic contention
    - N=512:  (32, 16) best → balance of parallelism
    - N=1024: (32, 64) best → more H parallelism
    - N=2048: (16, 64) best → reduce N atomic contention
    - N=4096: (16, 32) best → even smaller block_n
    - N=8192: (16-32, 32) → smaller blocks
    
    Key insight: FFN can use SMALLER block_n than fused_up because there's
    less atomic contention (no weight gradients, no linear projection work).
    """
    # Start with smaller block_n than forward
    if N <= 256:
        block_n, block_h = 16, 16
    elif N <= 512:
        block_n, block_h = 32, 16
    elif N <= 1024:
        block_n, block_h = 32, 64
    elif N <= 2048:
        block_n, block_h = 16, 64
    elif N <= 4096:
        block_n, block_h = 16, 32
    else:
        # Very large batches: smallest block_n to minimize atomic contention
        block_n, block_h = 32, 32
    
    # For many windows, reduce both dimensions to avoid register spills
    # in the forward recomputation
    if num_windows > 8:
        block_n = min(block_n, 16)
        block_h = min(block_h, 32)
    
    return block_n, block_h


@triton.jit
def template_gate_ffn_fwd(
    X, TEMPLATE, OUT,
    N, D, H,
    W: tl.constexpr,
    tau,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_WINDOWS: tl.constexpr,
):
    """
    Forward kernel for dendritic FFN.
    
    Computes: out = gate(x, template)
    
    Where gate is the template-matched weighted sum:
        dot = x_window · template_window  # raw dot products ARE the scores
        prob = softmax(|dot| / tau)
        out = sum_w(prob_w * dot_w) * (1 + ln(1 + |mean(dot)|))
    
    Output [N, H] is directly the gated template response, NOT gate * linear.
    
    Uses exp2 (base-2 exponentials) and bfloat16 for numerical stability.
    """
    # LOG2_E for converting exp(x) to exp2(x*log2(e))
    LOG2_E: tl.constexpr = 1.4426950408889634
    
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    
    n_mask = n_offsets < N
    h_mask = h_offsets < H
    
    total_span = NUM_WINDOWS * W
    w_idx = tl.arange(0, W)
    
    # Online softmax accumulators
    NEG_INF = -1e9
    max_abs_dot = tl.full((BLOCK_N, BLOCK_H), NEG_INF, dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.float32)
    weighted_sum = tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.float32)
    
    # Accumulator for mean computation
    dot_sum = tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.float32)
    
    for w_i in tl.static_range(NUM_WINDOWS):
        x_base = w_i * W
        
        # Load x_window: [BLOCK_N, W]
        x_ptrs = X + n_offsets[:, None] * D + (x_base + w_idx)[None, :]
        x_chunk = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
        
        # Load template_window: [BLOCK_H, W]
        tmpl_ptrs = TEMPLATE + h_offsets[:, None] * total_span + x_base + w_idx[None, :]
        tmpl_chunk = tl.load(tmpl_ptrs, mask=h_mask[:, None], other=0.0)
        
        # Dot product using bfloat16 for tensor core matmuls
        dot = tl.dot(x_chunk.to(tl.bfloat16), tl.trans(tmpl_chunk).to(tl.bfloat16)).to(tl.float32)
        
        # Absolute dot for softmax probabilities
        abs_dot = tl.abs(dot)
        # Scale by LOG2_E for exp2: exp(x/tau) = exp2(x*log2(e)/tau)
        scaled_abs_dot = abs_dot * (LOG2_E / tau)
        
        # Accumulate for mean
        dot_sum += dot
        
        # Online softmax update using exp2
        new_max = tl.maximum(max_abs_dot, scaled_abs_dot)
        exp_old = tl.exp2(max_abs_dot - new_max)
        exp_new = tl.exp2(scaled_abs_dot - new_max)
        
        weighted_sum = weighted_sum * exp_old + dot * exp_new
        sum_exp = sum_exp * exp_old + exp_new
        max_abs_dot = new_max
    
    # Compute mean and coactivation
    mean_dot = dot_sum / NUM_WINDOWS
    abs_mean_dot = tl.abs(mean_dot)
    
    # Final output = softmax_gate * coactivation
    # Use log2 for consistency: ln(x) = log2(x) / log2(e)
    softmax_gate = weighted_sum / (sum_exp + 1e-8)
    coactivation = 1.0 + tl.log2(1.0 + abs_mean_dot) / LOG2_E
    out = softmax_gate * coactivation
    
    out_ptrs = OUT + n_offsets[:, None] * H + h_offsets[None, :]
    tl.store(out_ptrs, out, mask=n_mask[:, None] & h_mask[None, :])


@triton.jit
def template_gate_ffn_bwd(
    GRAD_OUT,  # [N, H]
    X, TEMPLATE,  # inputs
    GRAD_X, GRAD_TEMPLATE,  # outputs
    N, D, H,
    W: tl.constexpr,
    tau,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_WINDOWS: tl.constexpr,
):
    """
    Backward kernel for dendritic FFN.
    
    Computes gradients for: out = softmax_gate * coactivation
    where softmax_gate = sum_w(prob_w * dot_w)
          coactivation = 1 + ln(1 + |mean(dot)|) (detached)
    
    No weights gradient since this kernel doesn't have a linear projection.
    
    Uses exp2 (base-2 exponentials) and bfloat16 for numerical stability.
    """
    # LOG2_E for converting exp(x) to exp2(x*log2(e))
    LOG2_E: tl.constexpr = 1.4426950408889634
    
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    
    n_mask = n_offsets < N
    h_mask = h_offsets < H
    
    total_span = NUM_WINDOWS * W
    w_idx = tl.arange(0, W)
    
    # Load grad_out: [BLOCK_N, BLOCK_H]
    grad_ptrs = GRAD_OUT + n_offsets[:, None] * H + h_offsets[None, :]
    grad_out = tl.load(grad_ptrs, mask=n_mask[:, None] & h_mask[None, :], other=0.0)
    
    # === First pass: recompute forward values ===
    NEG_INF = -1e9
    max_abs_dot = tl.full((BLOCK_N, BLOCK_H), NEG_INF, dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.float32)
    weighted_sum = tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.float32)
    dot_sum = tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.float32)
    
    # Process 2 windows per iteration for efficiency
    NUM_PAIRS: tl.constexpr = NUM_WINDOWS // 2
    
    for pair_i in tl.static_range(NUM_PAIRS):
        x_base0 = pair_i * 2 * W
        x_base1 = x_base0 + W
        
        # Window 0
        x_ptrs_0 = X + n_offsets[:, None] * D + (x_base0 + w_idx)[None, :]
        x_w0 = tl.load(x_ptrs_0, mask=n_mask[:, None], other=0.0)
        
        tmpl_ptrs_0 = TEMPLATE + h_offsets[:, None] * total_span + x_base0 + w_idx[None, :]
        tmpl_w0 = tl.load(tmpl_ptrs_0, mask=h_mask[:, None], other=0.0)
        
        # Raw dot products using bfloat16
        dot0 = tl.dot(x_w0.to(tl.bfloat16), tl.trans(tmpl_w0).to(tl.bfloat16)).to(tl.float32)
        # Scale by LOG2_E for exp2
        scaled0 = tl.abs(dot0) * (LOG2_E / tau)
        dot_sum += dot0
        
        new_max = tl.maximum(max_abs_dot, scaled0)
        exp_old = tl.exp2(max_abs_dot - new_max)
        exp_new = tl.exp2(scaled0 - new_max)
        weighted_sum = weighted_sum * exp_old + dot0 * exp_new
        sum_exp = sum_exp * exp_old + exp_new
        max_abs_dot = new_max
        
        # Window 1
        x_ptrs_1 = X + n_offsets[:, None] * D + (x_base1 + w_idx)[None, :]
        x_w1 = tl.load(x_ptrs_1, mask=n_mask[:, None], other=0.0)
        
        tmpl_ptrs_1 = TEMPLATE + h_offsets[:, None] * total_span + x_base1 + w_idx[None, :]
        tmpl_w1 = tl.load(tmpl_ptrs_1, mask=h_mask[:, None], other=0.0)
        
        # Raw dot products using bfloat16
        dot1 = tl.dot(x_w1.to(tl.bfloat16), tl.trans(tmpl_w1).to(tl.bfloat16)).to(tl.float32)
        # Scale by LOG2_E for exp2
        scaled1 = tl.abs(dot1) * (LOG2_E / tau)
        dot_sum += dot1
        
        new_max = tl.maximum(max_abs_dot, scaled1)
        exp_old = tl.exp2(max_abs_dot - new_max)
        exp_new = tl.exp2(scaled1 - new_max)
        weighted_sum = weighted_sum * exp_old + dot1 * exp_new
        sum_exp = sum_exp * exp_old + exp_new
        max_abs_dot = new_max
    
    # Compute forward values
    mean_dot = dot_sum / NUM_WINDOWS
    abs_mean_dot = tl.abs(mean_dot)
    softmax_gate = weighted_sum / (sum_exp + 1e-8)
    # Use log2 for consistency: ln(x) = log2(x) / log2(e)
    coactivation = 1.0 + tl.log2(1.0 + abs_mean_dot) / LOG2_E
    
    # Gradient of output w.r.t. softmax_gate (coactivation is detached)
    d_softmax_gate = grad_out * coactivation
    
    # Precompute for second pass - use LOG2_E scaling
    inv_sum_exp = 1.0 / (sum_exp + 1e-8)
    inv_tau_log2e = LOG2_E / tau
    
    # === Second pass: propagate gradients ===
    for pair_i in tl.static_range(NUM_PAIRS):
        x_base0 = pair_i * 2 * W
        x_base1 = x_base0 + W
        
        # ===== Window 0 =====
        x_ptrs_0 = X + n_offsets[:, None] * D + (x_base0 + w_idx)[None, :]
        x_w0 = tl.load(x_ptrs_0, mask=n_mask[:, None], other=0.0)
        
        tmpl_ptrs_0 = TEMPLATE + h_offsets[:, None] * total_span + x_base0 + w_idx[None, :]
        tmpl_w0 = tl.load(tmpl_ptrs_0, mask=h_mask[:, None], other=0.0)
        
        # Raw dot products using bfloat16
        dot0 = tl.dot(x_w0.to(tl.bfloat16), tl.trans(tmpl_w0).to(tl.bfloat16)).to(tl.float32)
        scaled0 = tl.abs(dot0) * inv_tau_log2e
        sign0 = tl.where(dot0 >= 0, 1.0, -1.0)
        
        prob0 = tl.exp2(scaled0 - max_abs_dot) * inv_sum_exp
        # Gradient w.r.t. dot with LOG2_E scaling
        d_dot0 = d_softmax_gate * prob0 * (1.0 + sign0 * (dot0 - softmax_gate) * inv_tau_log2e)
        
        # grad_template[0] using bfloat16
        d_tmpl0 = tl.dot(tl.trans(d_dot0).to(tl.bfloat16), x_w0.to(tl.bfloat16)).to(tl.float32)
        tl.atomic_add(GRAD_TEMPLATE + h_offsets[:, None] * total_span + x_base0 + w_idx[None, :], d_tmpl0, mask=h_mask[:, None])
        
        # grad_x[0] using bfloat16
        d_x0 = tl.dot(d_dot0.to(tl.bfloat16), tmpl_w0.to(tl.bfloat16)).to(tl.float32)
        tl.atomic_add(GRAD_X + n_offsets[:, None] * D + (x_base0 + w_idx)[None, :], d_x0, mask=n_mask[:, None])
        
        # ===== Window 1 =====
        x_ptrs_1 = X + n_offsets[:, None] * D + (x_base1 + w_idx)[None, :]
        x_w1 = tl.load(x_ptrs_1, mask=n_mask[:, None], other=0.0)
        
        tmpl_ptrs_1 = TEMPLATE + h_offsets[:, None] * total_span + x_base1 + w_idx[None, :]
        tmpl_w1 = tl.load(tmpl_ptrs_1, mask=h_mask[:, None], other=0.0)
        
        # Raw dot products using bfloat16
        dot1 = tl.dot(x_w1.to(tl.bfloat16), tl.trans(tmpl_w1).to(tl.bfloat16)).to(tl.float32)
        scaled1 = tl.abs(dot1) * inv_tau_log2e
        sign1 = tl.where(dot1 >= 0, 1.0, -1.0)
        
        prob1 = tl.exp2(scaled1 - max_abs_dot) * inv_sum_exp
        # Gradient w.r.t. dot
        d_dot1 = d_softmax_gate * prob1 * (1.0 + sign1 * (dot1 - softmax_gate) * inv_tau_log2e)
        
        # grad_template[1] using bfloat16
        d_tmpl1 = tl.dot(tl.trans(d_dot1).to(tl.bfloat16), x_w1.to(tl.bfloat16)).to(tl.float32)
        tl.atomic_add(GRAD_TEMPLATE + h_offsets[:, None] * total_span + x_base1 + w_idx[None, :], d_tmpl1, mask=h_mask[:, None])
        
        # grad_x[1] using bfloat16
        d_x1 = tl.dot(d_dot1.to(tl.bfloat16), tmpl_w1.to(tl.bfloat16)).to(tl.float32)
        tl.atomic_add(GRAD_X + n_offsets[:, None] * D + (x_base1 + w_idx)[None, :], d_x1, mask=n_mask[:, None])


class TemplateGateFFNFunction(Function):
    """Autograd function for dendritic FFN (template gate without linear projection)."""
    
    @staticmethod
    def forward(ctx, x, template_flat, tau, W, num_windows, block_n, block_h, H):
        """
        Forward pass for dendritic FFN.
        
        Args:
            x: [N, D] input
            template_flat: [H, num_windows*W] pre-flattened template
            tau: temperature for softmax
            H: output dimension (equals hidden_dim)
        
        Returns:
            out: [N, H] gated template response
        """
        N, D = x.shape
        
        out = torch.empty(N, H, device=x.device, dtype=x.dtype)
        
        grid = (triton.cdiv(N, block_n), triton.cdiv(H, block_h))
        
        template_gate_ffn_fwd[grid](
            x, template_flat, out,
            N, D, H, W, tau,
            BLOCK_N=block_n, BLOCK_H=block_h, NUM_WINDOWS=num_windows
        )
        
        ctx.save_for_backward(x, template_flat)
        ctx.tau = tau
        ctx.W = W
        ctx.num_windows = num_windows
        ctx.block_n = block_n
        ctx.block_h = block_h
        ctx.H = H
        ctx.D = D
        ctx.N = N
        
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        x, template_flat = ctx.saved_tensors
        N = ctx.N
        D = ctx.D
        H = ctx.H
        
        grad_x = torch.zeros_like(x)
        grad_template_flat = torch.zeros_like(template_flat)
        
        # Get optimal block sizes for backward
        bwd_block_n, bwd_block_h = _get_ffn_bwd_block_sizes(N, H, ctx.W, ctx.num_windows)
        
        grid = (triton.cdiv(N, bwd_block_n), triton.cdiv(H, bwd_block_h))
        
        template_gate_ffn_bwd[grid](
            grad_out.contiguous(),
            x, template_flat,
            grad_x, grad_template_flat,
            N, D, H, ctx.W, ctx.tau,
            BLOCK_N=bwd_block_n, BLOCK_H=bwd_block_h, NUM_WINDOWS=ctx.num_windows
        )
        
        # Return gradients for all forward inputs (None for non-tensor args)
        return grad_x, grad_template_flat, None, None, None, None, None, None


def template_gate_ffn_flat(x, template_flat, H, tau=1.0, W=64,
                            num_windows=None, block_n=None, block_h=None):
    """
    Dendritic FFN with pre-flattened template.
    
    This is the main entry point for the dendritic FFN kernel.
    
    Args:
        x: Input tensor [N, D]
        template_flat: Pre-flattened template [H, num_windows * W]
        H: Output dimension
        tau: Temperature for softmax
        W: Window size
        num_windows: Number of windows (D // W if not provided)
        block_n: Block size for N dimension (auto-tuned if None)
        block_h: Block size for H dimension (auto-tuned if None)
    
    Returns:
        Output tensor [N, H] = gated template response
    """
    N, D = x.shape
    if num_windows is None:
        num_windows = D // W
    
    # Auto-tune block sizes if not provided
    if block_n is None or block_h is None:
        block_n, block_h = _get_ffn_fwd_block_sizes(N, H, W, num_windows)
    
    return TemplateGateFFNFunction.apply(
        x, template_flat, tau, W, num_windows, block_n, block_h, H
    )


def template_gate_ffn(x, template, tau=1.0, W=64, block_n=None, block_h=None):
    """
    Dendritic FFN with 3D template.
    
    Convenience wrapper that accepts 3D template and flattens it.
    
    Args:
        x: Input tensor [N, D]
        template: Template tensor [H, num_windows, W]
        tau: Temperature for softmax
        W: Window size
        block_n: Block size for N dimension (auto-tuned if None)
        block_h: Block size for H dimension (auto-tuned if None)
    
    Returns:
        Output tensor [N, H] = gated template response
    """
    H, num_windows, _ = template.shape
    N = x.shape[0]
    template_flat = template.view(H, -1).contiguous()
    
    # Auto-tune block sizes if not provided
    if block_n is None or block_h is None:
        block_n, block_h = _get_ffn_fwd_block_sizes(N, H, W, num_windows)
    
    return TemplateGateFFNFunction.apply(
        x, template_flat, tau, W, num_windows, block_n, block_h, H
    )


if __name__ == "__main__":
    import math
    
    device = 'cuda'
    D, H, W = 512, 2048, 64
    num_windows = D // W
    tau = 1.0
    
    print("=" * 70)
    print("Dendritic FFN Kernel - Template-Gated Direct Projection")
    print("=" * 70)
    
    # Test correctness with gradients
    print("\n--- Basic Gradient Test ---")
    x = torch.randn(64, D, device=device).requires_grad_(True)
    template = (torch.randn(H, num_windows, W, device=device) * 0.02).requires_grad_(True)
    
    # Retain grads for non-leaf tensors
    x.retain_grad()
    template.retain_grad()
    
    out = template_gate_ffn(x, template, tau=tau, W=W)
    loss = out.sum()
    loss.backward()
    
    print(f"Forward shape: {out.shape}")
    print(f"x.grad: {x.grad is not None}, norm={x.grad.norm().item():.4f}")
    print(f"template.grad: {template.grad is not None}, norm={template.grad.norm().item():.4f}")
    print(f"Gradients computed successfully!")
    
    # === Numerical Gradient Check ===
    print("\n--- Numerical Gradient Check ---")
    
    def check_numerical_grad(name, param, compute_out, eps=1e-4, num_checks=5):
        """Check gradients numerically for a subset of elements."""
        param_flat = param.view(-1)
        errors = []
        
        for i in range(min(num_checks, param_flat.numel())):
            idx = torch.randint(0, param_flat.numel(), (1,)).item()
            
            # f(x + eps)
            param_flat.data[idx] += eps
            out_plus = compute_out().sum()
            param_flat.data[idx] -= eps
            
            # f(x - eps)
            param_flat.data[idx] -= eps
            out_minus = compute_out().sum()
            param_flat.data[idx] += eps
            
            # Numerical gradient
            num_grad = (out_plus - out_minus) / (2 * eps)
            
            # Analytical gradient
            ana_grad = param.grad.view(-1)[idx].item()
            
            # Relative error
            if abs(num_grad) > 1e-7 or abs(ana_grad) > 1e-7:
                rel_err = abs(num_grad - ana_grad) / (max(abs(num_grad), abs(ana_grad)) + 1e-8)
                errors.append(rel_err)
        
        if errors:
            max_err = max(errors)
            avg_err = sum(errors) / len(errors)
            print(f"  {name}: max_rel_err={max_err:.2e}, avg_rel_err={avg_err:.2e}")
            return max_err < 0.1  # Allow 10% relative error due to fp16 in kernel
        return True
    
    # Reset grads
    x = torch.randn(16, D, device=device).requires_grad_(True)
    template = (torch.randn(H, num_windows, W, device=device) * 0.02).requires_grad_(True)
    
    def compute_out():
        return template_gate_ffn(x, template, tau=tau, W=W)
    
    # Compute analytical gradients
    out = compute_out()
    out.sum().backward()
    
    x_ok = check_numerical_grad("x", x, compute_out)
    template_ok = check_numerical_grad("template", template, compute_out)
    
    if x_ok and template_ok:
        print("  ✓ Numerical gradient check PASSED")
    else:
        print("  ✗ Numerical gradient check FAILED")
    
    # === Gradient Flow Analysis ===
    print("\n--- Gradient Flow Analysis ---")
    print("Testing gradient magnitude across input scales:")
    
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        x = (torch.randn(64, D, device=device) * scale).requires_grad_(True)
        template = (torch.randn(H, num_windows, W, device=device) * 0.02).requires_grad_(True)
        
        out = template_gate_ffn(x, template, tau=tau, W=W)
        out.sum().backward()
        
        print(f"  scale={scale:.1f}: x.grad_norm={x.grad.norm():.4f}, tmpl.grad_norm={template.grad.norm():.4f}")
    
    # === Performance Comparison ===
    print("\n--- Performance Comparison vs Swish FFN ---")
    print("(Swish FFN = x @ W.T followed by swish activation)")
    
    import time
    
    class SwishFFN(torch.nn.Module):
        """Simple Swish FFN for comparison."""
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = torch.nn.Linear(in_features, out_features, bias=False)
            
        def forward(self, x):
            h = self.linear(x)
            return h * torch.sigmoid(h)  # Swish
    
    swish_ffn = SwishFFN(D, H).to(device)
    
    # Warmup
    x = torch.randn(256, D, device=device)
    template_flat = torch.randn(H, num_windows * W, device=device) * 0.02
    for _ in range(10):
        _ = template_gate_ffn_flat(x, template_flat, H, tau=tau, W=W)
        _ = swish_ffn(x)
    torch.cuda.synchronize()
    
    print(f"\nD={D}, H={H}, W={W}, num_windows={num_windows}")
    print("-" * 60)
    
    for N in [256, 512, 1024, 2048, 4096, 8192]:
        x = torch.randn(N, D, device=device).requires_grad_(True)
        x_swish = x.detach().clone().requires_grad_(True)
        template_flat = (torch.randn(H, num_windows * W, device=device) * 0.02).requires_grad_(True)
        
        # Forward timing - Dendritic FFN
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            out = template_gate_ffn_flat(x, template_flat, H, tau=tau, W=W)
        torch.cuda.synchronize()
        dendritic_fwd = (time.perf_counter() - start) / 50 * 1000
        
        # Forward timing - Swish FFN
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            out_swish = swish_ffn(x_swish)
        torch.cuda.synchronize()
        swish_fwd = (time.perf_counter() - start) / 50 * 1000
        
        # Backward timing - Dendritic FFN
        out = template_gate_ffn_flat(x, template_flat, H, tau=tau, W=W)
        grad = torch.ones_like(out)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            x.grad = None
            template_flat.grad = None
            out.backward(grad, retain_graph=True)
        torch.cuda.synchronize()
        dendritic_bwd = (time.perf_counter() - start) / 50 * 1000
        
        # Backward timing - Swish FFN
        out_swish = swish_ffn(x_swish)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            x_swish.grad = None
            swish_ffn.linear.weight.grad = None
            out_swish.backward(grad, retain_graph=True)
        torch.cuda.synchronize()
        swish_bwd = (time.perf_counter() - start) / 50 * 1000
        
        fwd_ratio = swish_fwd / dendritic_fwd
        bwd_ratio = swish_bwd / dendritic_bwd
        total_ratio = (swish_fwd + swish_bwd) / (dendritic_fwd + dendritic_bwd)
        
        print(f"N={N:5d}: Dendritic [fwd={dendritic_fwd:.3f}ms, bwd={dendritic_bwd:.3f}ms] "
              f"Swish [fwd={swish_fwd:.3f}ms, bwd={swish_bwd:.3f}ms] "
              f"Ratio [fwd={fwd_ratio:.2f}x, bwd={bwd_ratio:.2f}x, tot={total_ratio:.2f}x]")
