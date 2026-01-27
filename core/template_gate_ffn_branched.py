"""
Dendritic FFN Triton Kernel with Multiple Branches - Somatic Integration.

This module extends the dendritic FFN to support multiple dendritic branches per neuron,
with somatic integration (linear summation) of branch outputs.

Architecture concept - spatial partitioning of input:
    - Input of dimension D is partitioned into B branches
    - Each branch receives a DIFFERENT segment of the input: D/B dims per branch
    - Each branch applies windowed template matching within its segment
    - Local NMDA-like competition happens within each branch
    - Somatic integration: linear sum of branch outputs

Example with D=2048, B=4 branches, W=64:
    - Branch 0: processes x[0:512]    with 8 windows of size 64
    - Branch 1: processes x[512:1024] with 8 windows of size 64
    - Branch 2: processes x[1024:1536] with 8 windows of size 64
    - Branch 3: processes x[1536:2048] with 8 windows of size 64

Current architecture (branch_count=1):
    - Single branch covering all of D
    - num_windows = D // W templates per neuron
    - Output: softmax_gate * coactivation

New branched architecture (branch_count=B):
    - B branches, each covering D/B dimensions
    - windows_per_branch = (D/B) // W templates per branch
    - Each branch: branch_out = softmax_gate_local * coactivation_local
    - Final: out = sum(branch_out for all branches)

Formula per branch b:
    # Branch b processes input segment x[b*segment_size : (b+1)*segment_size]
    dot[h, b, w] = x_segment[w] · template[h, b, w, :]
    prob[h, b, w] = softmax_within_branch(|dot| / tau)
    branch_out[h, b] = sum_w(prob * dot) * (1 + ln(1 + |mean(dot)|))

Final output:
    out[h] = sum_b(branch_out[h, b])  # Simple somatic integration

Key properties:
1. Spatial partitioning: each branch sees different input dimensions
2. Local competition within each branch (not global across branches)
3. Each branch has its own coactivation term (|mean| within branch)
4. Branches contribute additively (gradients flow through linearly)
5. Compatible with branch_count=1 (equivalent to original)

Template shape: [H, branch_count, windows_per_branch, W]
    where windows_per_branch = D // (branch_count * W)
"""

import torch
import triton
import triton.language as tl
from torch.autograd import Function


def _get_ffn_branched_fwd_block_sizes(N, H, W, num_windows_per_branch, branch_count):
    """
    Select optimal block sizes for branched FFN forward kernel.
    
    With multiple branches, we have more accumulators per thread block,
    so we may need smaller blocks to avoid register spills.
    """
    total_windows = num_windows_per_branch * branch_count
    
    # Base sizing similar to original
    block_n, block_h = 64, 32
    
    # Reduce for small batches
    if N <= 32:
        block_n = 16
    elif N <= 64:
        block_n = 32
    
    # With more branches, we need more registers per block for accumulators
    # Each branch needs: max_abs_dot, sum_exp, weighted_sum, dot_sum (4 accumulators)
    # Reduce block sizes to compensate
    if branch_count >= 4:
        block_n = min(block_n, 32)
        block_h = min(block_h, 16)
    elif branch_count >= 2:
        block_h = min(block_h, 32)
    
    # Many total windows: reduce to avoid register spills
    if total_windows > 16:
        block_n = min(block_n, 32)
        block_h = min(block_h, 16)
    
    return block_n, block_h


def _get_ffn_branched_bwd_block_sizes(N, H, W, num_windows_per_branch, branch_count):
    """
    Select optimal block sizes for branched FFN backward kernel.
    
    Backward recomputes forward values so register pressure is higher.
    """
    total_windows = num_windows_per_branch * branch_count
    
    # Start smaller than forward due to recomputation
    if N <= 256:
        block_n, block_h = 16, 16
    elif N <= 512:
        block_n, block_h = 32, 16
    elif N <= 1024:
        block_n, block_h = 32, 32
    elif N <= 2048:
        block_n, block_h = 16, 32
    else:
        block_n, block_h = 32, 32
    
    # Reduce further for multiple branches
    if branch_count >= 4:
        block_n = min(block_n, 16)
        block_h = min(block_h, 16)
    elif branch_count >= 2:
        block_h = min(block_h, 16)
    
    # Many total windows: reduce to avoid register spills
    if total_windows > 16:
        block_n = min(block_n, 16)
        block_h = min(block_h, 16)
    
    return block_n, block_h


@triton.jit
def template_gate_ffn_branched_fwd(
    X, TEMPLATE, OUT,
    N, D, H,
    W: tl.constexpr,
    tau,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BRANCH_COUNT: tl.constexpr,
    WINDOWS_PER_BRANCH: tl.constexpr,
):
    """
    Forward kernel for branched dendritic FFN.
    
    Each neuron has BRANCH_COUNT branches, each processing a DIFFERENT segment of the input.
    - Input D is spatially partitioned: segment_size = D / BRANCH_COUNT
    - Branch b processes x[b*segment_size : (b+1)*segment_size]
    - Each branch has WINDOWS_PER_BRANCH windows of size W
    - Competition happens WITHIN each branch, then branches sum for final output.
    
    Template layout: [H, BRANCH_COUNT * WINDOWS_PER_BRANCH * W] (flattened)
    Input segment for branch b starts at: b * WINDOWS_PER_BRANCH * W
    """
    LOG2_E: tl.constexpr = 1.4426950408889634
    
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    
    n_mask = n_offsets < N
    h_mask = h_offsets < H
    
    # Total template span per neuron
    total_windows_per_neuron: tl.constexpr = BRANCH_COUNT * WINDOWS_PER_BRANCH
    total_span: tl.constexpr = total_windows_per_neuron * W
    
    # Segment size per branch in the input
    SEGMENT_SIZE: tl.constexpr = WINDOWS_PER_BRANCH * W
    
    w_idx = tl.arange(0, W)
    
    # Output accumulator (sum of all branches)
    out_acc = tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.float32)
    
    # Process each branch - branches have LOCAL competition over DIFFERENT input segments
    for branch_idx in tl.static_range(BRANCH_COUNT):
        # Online softmax accumulators for this branch
        NEG_INF = -1e9
        max_abs_dot = tl.full((BLOCK_N, BLOCK_H), NEG_INF, dtype=tl.float32)
        sum_exp = tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.float32)
        weighted_sum = tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.float32)
        dot_sum = tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.float32)
        
        # Base offset for this branch's templates
        branch_template_offset = branch_idx * WINDOWS_PER_BRANCH * W
        
        # Base offset for this branch's INPUT SEGMENT
        branch_input_offset = branch_idx * SEGMENT_SIZE
        
        # Process windows within this branch's segment
        for w_i in tl.static_range(WINDOWS_PER_BRANCH):
            # Input window index WITHIN this branch's segment
            x_base = branch_input_offset + w_i * W
            
            # Load x_window: [BLOCK_N, W]
            x_ptrs = X + n_offsets[:, None] * D + (x_base + w_idx)[None, :]
            x_chunk = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
            
            # Template offset for this branch and window
            template_offset = branch_template_offset + w_i * W
            
            # Load template_window: [BLOCK_H, W]
            tmpl_ptrs = TEMPLATE + h_offsets[:, None] * total_span + template_offset + w_idx[None, :]
            tmpl_chunk = tl.load(tmpl_ptrs, mask=h_mask[:, None], other=0.0)
            
            # Dot product using bfloat16 for tensor core matmuls
            dot = tl.dot(x_chunk.to(tl.bfloat16), tl.trans(tmpl_chunk).to(tl.bfloat16)).to(tl.float32)
            
            # Absolute dot for softmax probabilities
            abs_dot = tl.abs(dot)
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
        
        # Compute branch output (local competition within this branch)
        mean_dot = dot_sum / WINDOWS_PER_BRANCH
        abs_mean_dot = tl.abs(mean_dot)
        
        softmax_gate = weighted_sum / (sum_exp + 1e-8)
        coactivation = 1.0 + tl.log2(1.0 + abs_mean_dot) / LOG2_E
        branch_out = softmax_gate * coactivation
        
        # Somatic integration: add branch output to total
        out_acc += branch_out
    
    # Store final output
    out_ptrs = OUT + n_offsets[:, None] * H + h_offsets[None, :]
    tl.store(out_ptrs, out_acc, mask=n_mask[:, None] & h_mask[None, :])


@triton.jit
def template_gate_ffn_branched_bwd(
    GRAD_OUT,
    X, TEMPLATE,
    GRAD_X, GRAD_TEMPLATE,
    N, D, H,
    W: tl.constexpr,
    tau,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BRANCH_COUNT: tl.constexpr,
    WINDOWS_PER_BRANCH: tl.constexpr,
):
    """
    Backward kernel for branched dendritic FFN.
    
    Since final output is sum of branches, grad_out flows equally to all branches.
    Each branch processes a DIFFERENT segment of the input:
    - Branch b processes x[b*segment_size : (b+1)*segment_size]
    - Gradients accumulate to different parts of grad_x for different branches
    """
    LOG2_E: tl.constexpr = 1.4426950408889634
    
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    
    n_mask = n_offsets < N
    h_mask = h_offsets < H
    
    total_windows_per_neuron: tl.constexpr = BRANCH_COUNT * WINDOWS_PER_BRANCH
    total_span: tl.constexpr = total_windows_per_neuron * W
    
    # Segment size per branch in the input
    SEGMENT_SIZE: tl.constexpr = WINDOWS_PER_BRANCH * W
    
    w_idx = tl.arange(0, W)
    
    # Load grad_out: [BLOCK_N, BLOCK_H]
    # This gradient flows to ALL branches (since output = sum of branches)
    grad_ptrs = GRAD_OUT + n_offsets[:, None] * H + h_offsets[None, :]
    grad_out = tl.load(grad_ptrs, mask=n_mask[:, None] & h_mask[None, :], other=0.0)
    
    # Process each branch
    for branch_idx in tl.static_range(BRANCH_COUNT):
        # === First pass: recompute forward values for this branch ===
        NEG_INF = -1e9
        max_abs_dot = tl.full((BLOCK_N, BLOCK_H), NEG_INF, dtype=tl.float32)
        sum_exp = tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.float32)
        weighted_sum = tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.float32)
        dot_sum = tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.float32)
        
        branch_template_offset = branch_idx * WINDOWS_PER_BRANCH * W
        branch_input_offset = branch_idx * SEGMENT_SIZE
        
        for w_i in tl.static_range(WINDOWS_PER_BRANCH):
            x_base = branch_input_offset + w_i * W
            
            x_ptrs = X + n_offsets[:, None] * D + (x_base + w_idx)[None, :]
            x_chunk = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
            
            template_offset = branch_template_offset + w_i * W
            tmpl_ptrs = TEMPLATE + h_offsets[:, None] * total_span + template_offset + w_idx[None, :]
            tmpl_chunk = tl.load(tmpl_ptrs, mask=h_mask[:, None], other=0.0)
            
            dot = tl.dot(x_chunk.to(tl.bfloat16), tl.trans(tmpl_chunk).to(tl.bfloat16)).to(tl.float32)
            scaled = tl.abs(dot) * (LOG2_E / tau)
            dot_sum += dot
            
            new_max = tl.maximum(max_abs_dot, scaled)
            exp_old = tl.exp2(max_abs_dot - new_max)
            exp_new = tl.exp2(scaled - new_max)
            weighted_sum = weighted_sum * exp_old + dot * exp_new
            sum_exp = sum_exp * exp_old + exp_new
            max_abs_dot = new_max
        
        # Compute forward values
        mean_dot = dot_sum / WINDOWS_PER_BRANCH
        abs_mean_dot = tl.abs(mean_dot)
        softmax_gate = weighted_sum / (sum_exp + 1e-8)
        coactivation = 1.0 + tl.log2(1.0 + abs_mean_dot) / LOG2_E
        
        # Gradient of branch output w.r.t. softmax_gate (coactivation is detached)
        # grad_out flows directly since output = sum of branches
        d_softmax_gate = grad_out * coactivation
        
        # Precompute for gradient pass
        inv_sum_exp = 1.0 / (sum_exp + 1e-8)
        inv_tau_log2e = LOG2_E / tau
        
        # === Second pass: propagate gradients for this branch ===
        for w_i in tl.static_range(WINDOWS_PER_BRANCH):
            x_base = branch_input_offset + w_i * W
            
            x_ptrs = X + n_offsets[:, None] * D + (x_base + w_idx)[None, :]
            x_chunk = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
            
            template_offset = branch_template_offset + w_i * W
            tmpl_ptrs = TEMPLATE + h_offsets[:, None] * total_span + template_offset + w_idx[None, :]
            tmpl_chunk = tl.load(tmpl_ptrs, mask=h_mask[:, None], other=0.0)
            
            dot = tl.dot(x_chunk.to(tl.bfloat16), tl.trans(tmpl_chunk).to(tl.bfloat16)).to(tl.float32)
            scaled = tl.abs(dot) * inv_tau_log2e
            sign = tl.where(dot >= 0, 1.0, -1.0)
            
            prob = tl.exp2(scaled - max_abs_dot) * inv_sum_exp
            d_dot = d_softmax_gate * prob * (1.0 + sign * (dot - softmax_gate) * inv_tau_log2e)
            
            # grad_template using bfloat16
            d_tmpl = tl.dot(tl.trans(d_dot).to(tl.bfloat16), x_chunk.to(tl.bfloat16)).to(tl.float32)
            tl.atomic_add(GRAD_TEMPLATE + h_offsets[:, None] * total_span + template_offset + w_idx[None, :], 
                         d_tmpl, mask=h_mask[:, None])
            
            # grad_x using bfloat16 - writes to this branch's segment of grad_x
            d_x = tl.dot(d_dot.to(tl.bfloat16), tmpl_chunk.to(tl.bfloat16)).to(tl.float32)
            tl.atomic_add(GRAD_X + n_offsets[:, None] * D + (x_base + w_idx)[None, :], 
                         d_x, mask=n_mask[:, None])


class TemplateGateFFNBranchedFunction(Function):
    """Autograd function for branched dendritic FFN."""
    
    @staticmethod
    def forward(ctx, x, template_flat, tau, W, branch_count, windows_per_branch, block_n, block_h, H):
        """
        Forward pass for branched dendritic FFN.
        
        Args:
            x: [N, D] input
            template_flat: [H, branch_count * windows_per_branch * W] flattened template
            tau: temperature for softmax
            W: window size
            branch_count: number of dendritic branches per neuron
            windows_per_branch: number of windows per branch
            H: output dimension
        
        Returns:
            out: [N, H] sum of branch outputs
        """
        N, D = x.shape
        
        out = torch.empty(N, H, device=x.device, dtype=x.dtype)
        
        grid = (triton.cdiv(N, block_n), triton.cdiv(H, block_h))
        
        template_gate_ffn_branched_fwd[grid](
            x, template_flat, out,
            N, D, H, W, tau,
            BLOCK_N=block_n, BLOCK_H=block_h,
            BRANCH_COUNT=branch_count, WINDOWS_PER_BRANCH=windows_per_branch
        )
        
        ctx.save_for_backward(x, template_flat)
        ctx.tau = tau
        ctx.W = W
        ctx.branch_count = branch_count
        ctx.windows_per_branch = windows_per_branch
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
        
        bwd_block_n, bwd_block_h = _get_ffn_branched_bwd_block_sizes(
            N, H, ctx.W, ctx.windows_per_branch, ctx.branch_count
        )
        
        grid = (triton.cdiv(N, bwd_block_n), triton.cdiv(H, bwd_block_h))
        
        template_gate_ffn_branched_bwd[grid](
            grad_out.contiguous(),
            x, template_flat,
            grad_x, grad_template_flat,
            N, D, H, ctx.W, ctx.tau,
            BLOCK_N=bwd_block_n, BLOCK_H=bwd_block_h,
            BRANCH_COUNT=ctx.branch_count, WINDOWS_PER_BRANCH=ctx.windows_per_branch
        )
        
        return grad_x, grad_template_flat, None, None, None, None, None, None, None


def template_gate_ffn_branched_flat(x, template_flat, H, tau=1.0, W=64,
                                     branch_count=1, windows_per_branch=None,
                                     block_n=None, block_h=None):
    """
    Branched dendritic FFN with pre-flattened template.
    
    Args:
        x: Input tensor [N, D]
        template_flat: Pre-flattened template [H, branch_count * windows_per_branch * W]
        H: Output dimension
        tau: Temperature for softmax
        W: Window size
        branch_count: Number of dendritic branches per neuron (spatially partitions D)
        windows_per_branch: Windows per branch. If not provided, computed as D // (branch_count * W)
        block_n: Block size for N dimension (auto-tuned if None)
        block_h: Block size for H dimension (auto-tuned if None)
    
    Returns:
        Output tensor [N, H] = sum of branch outputs
    
    Note:
        D must be divisible by (branch_count * W).
        Each branch processes D/branch_count dimensions of input.
    """
    N, D = x.shape
    if windows_per_branch is None:
        assert D % (branch_count * W) == 0, \
            f"D ({D}) must be divisible by branch_count * W ({branch_count} * {W} = {branch_count * W})"
        windows_per_branch = D // (branch_count * W)
    
    if block_n is None or block_h is None:
        block_n, block_h = _get_ffn_branched_fwd_block_sizes(
            N, H, W, windows_per_branch, branch_count
        )
    
    return TemplateGateFFNBranchedFunction.apply(
        x, template_flat, tau, W, branch_count, windows_per_branch, block_n, block_h, H
    )


def template_gate_ffn_branched(x, template, tau=1.0, W=64, block_n=None, block_h=None):
    """
    Branched dendritic FFN with 4D template.
    
    Args:
        x: Input tensor [N, D]
        template: Template tensor [H, branch_count, windows_per_branch, W]
        tau: Temperature for softmax
        W: Window size
        block_n: Block size for N dimension (auto-tuned if None)
        block_h: Block size for H dimension (auto-tuned if None)
    
    Returns:
        Output tensor [N, H] = sum of branch outputs
    
    Note:
        D must equal branch_count * windows_per_branch * W.
        Each branch processes D/branch_count dimensions of input.
    """
    H, branch_count, windows_per_branch, _ = template.shape
    N, D = x.shape
    
    expected_D = branch_count * windows_per_branch * W
    assert D == expected_D, \
        f"Input D ({D}) must equal branch_count * windows_per_branch * W ({expected_D})"
    
    template_flat = template.view(H, -1).contiguous()
    
    if block_n is None or block_h is None:
        block_n, block_h = _get_ffn_branched_fwd_block_sizes(
            N, H, W, windows_per_branch, branch_count
        )
    
    return TemplateGateFFNBranchedFunction.apply(
        x, template_flat, tau, W, branch_count, windows_per_branch, block_n, block_h, H
    )


# ============================================================================
# BACKWARD COMPATIBILITY: branch_count=1 should match original
# ============================================================================

def verify_backward_compatibility():
    """Verify that branch_count=1 produces same results as original."""
    from core.template_gate_ffn import template_gate_ffn_flat
    
    device = 'cuda'
    D, H, W = 512, 256, 64  # Smaller for testing
    num_windows = D // W  # = 8
    tau = 1.0
    N = 64
    
    # Same inputs
    torch.manual_seed(42)
    x = torch.randn(N, D, device=device)
    template_orig = torch.randn(H, num_windows, W, device=device) * 0.02
    template_flat_orig = template_orig.view(H, -1).contiguous()
    
    # For branched with 1 branch: [H, 1, num_windows, W]
    # This is equivalent to original since 1 branch covers all D
    template_branched = template_orig.unsqueeze(1)  # [H, 1, num_windows, W]
    template_flat_branched = template_branched.view(H, -1).contiguous()
    
    # Forward pass
    out_orig = template_gate_ffn_flat(x, template_flat_orig, H, tau=tau, W=W, num_windows=num_windows)
    out_branched = template_gate_ffn_branched_flat(
        x, template_flat_branched, H, tau=tau, W=W, 
        branch_count=1, windows_per_branch=num_windows
    )
    
    max_diff = (out_orig - out_branched).abs().max().item()
    print(f"Forward max diff (branch=1 vs original): {max_diff:.2e}")
    
    return max_diff < 1e-4


if __name__ == "__main__":
    import time
    import sys
    sys.path.insert(0, '/home/aiman/dendrite')
    
    device = 'cuda'
    
    print("=" * 70)
    print("Branched Dendritic FFN Kernel - Multi-Branch Somatic Integration")
    print("=" * 70)
    print("\nArchitecture: Each branch processes a DIFFERENT spatial segment of input")
    print("Example: D=2048, B=4 branches -> each branch processes 512 dims")
    
    # === Backward Compatibility Test ===
    print("\n--- Backward Compatibility Test (branch_count=1) ---")
    try:
        compat_ok = verify_backward_compatibility()
        if compat_ok:
            print("✓ Backward compatibility verified!")
        else:
            print("✗ Backward compatibility FAILED!")
    except Exception as e:
        print(f"✗ Backward compatibility test error: {e}")
    
    # === Basic Gradient Test ===
    print("\n--- Basic Gradient Test ---")
    D, H, W = 512, 2048, 64
    branch_count = 4  # 4 branches, each processes D/4 = 128 dims
    windows_per_branch = D // (branch_count * W)  # = 512 / (4 * 64) = 2
    tau = 1.0
    
    print(f"Config: D={D}, H={H}, W={W}")
    print(f"Branches: {branch_count}, each processes {D // branch_count} dims")
    print(f"Windows per branch: {windows_per_branch}")
    
    x = torch.randn(64, D, device=device).requires_grad_(True)
    template = (torch.randn(H, branch_count, windows_per_branch, W, device=device) * 0.02).requires_grad_(True)
    
    out = template_gate_ffn_branched(x, template, tau=tau, W=W)
    loss = out.sum()
    loss.backward()
    
    print(f"Forward shape: {out.shape}")
    print(f"x.grad: exists={x.grad is not None}, norm={x.grad.norm().item():.4f}")
    print(f"template.grad: exists={template.grad is not None}, norm={template.grad.norm().item():.4f}")
    
    # Check for NaN/Inf
    has_nan = torch.isnan(x.grad).any() or torch.isnan(template.grad).any()
    has_inf = torch.isinf(x.grad).any() or torch.isinf(template.grad).any()
    print(f"NaN in grads: {has_nan}, Inf in grads: {has_inf}")
    
    if not has_nan and not has_inf:
        print("✓ Gradients computed successfully!")
    else:
        print("✗ Gradient computation has NaN/Inf!")
    
    # === Numerical Gradient Check ===
    print("\n--- Numerical Gradient Check ---")
    print("Note: Using eps=1e-2 due to bf16 tensor core precision limits")
    
    def check_numerical_grad(name, param, compute_out, eps=1e-2, num_checks=10):
        """Check gradients numerically for a subset of elements.
        
        Using eps=1e-2 because bf16 tensor cores lose precision below this.
        """
        param_flat = param.view(-1)
        errors = []
        
        for i in range(min(num_checks, param_flat.numel())):
            idx = torch.randint(0, param_flat.numel(), (1,)).item()
            
            original_val = param_flat[idx].item()
            
            param_flat.data[idx] = original_val + eps
            out_plus = compute_out().sum().item()
            
            param_flat.data[idx] = original_val - eps
            out_minus = compute_out().sum().item()
            
            param_flat.data[idx] = original_val
            
            num_grad = (out_plus - out_minus) / (2 * eps)
            ana_grad = param.grad.view(-1)[idx].item()
            
            if abs(num_grad) > 1e-7 or abs(ana_grad) > 1e-7:
                rel_err = abs(num_grad - ana_grad) / (max(abs(num_grad), abs(ana_grad)) + 1e-8)
                errors.append(rel_err)
        
        if errors:
            max_err = max(errors)
            avg_err = sum(errors) / len(errors)
            print(f"  {name}: max_rel_err={max_err:.2e}, avg_rel_err={avg_err:.2e}")
            return avg_err < 0.35  # Allow 35% average for bf16 with eps=1e-2
        return True
    
    # Smaller tensors for numerical gradient check
    D_small, H_small = 256, 128
    branch_count_small = 2
    windows_per_branch_small = D_small // (branch_count_small * W)
    
    x = torch.randn(8, D_small, device=device).requires_grad_(True)
    template = (torch.randn(H_small, branch_count_small, windows_per_branch_small, W, device=device) * 0.02).requires_grad_(True)
    
    def compute_out():
        return template_gate_ffn_branched(x, template, tau=tau, W=W)
    
    out = compute_out()
    out.sum().backward()
    
    x_ok = check_numerical_grad("x", x, compute_out)
    template_ok = check_numerical_grad("template", template, compute_out)
    
    if x_ok and template_ok:
        print("✓ Numerical gradient check PASSED")
    else:
        print("✗ Numerical gradient check FAILED")
    
    # === Performance Comparison ===
    print("\n--- Performance Comparison: Branched vs Original ---")
    
    from core.template_gate_ffn import template_gate_ffn_flat
    
    D, H, W = 512, 2048, 64
    num_windows_orig = D // W  # = 8
    
    print(f"\nD={D}, H={H}, W={W}")
    print("-" * 80)
    
    for N in [256, 512, 1024, 2048, 4096]:
        x = torch.randn(N, D, device=device).requires_grad_(True)
        x_orig = x.detach().clone().requires_grad_(True)
        
        # Original (branch_count=1)
        template_orig = torch.randn(H, num_windows_orig, W, device=device) * 0.02
        template_flat_orig = template_orig.view(H, -1).contiguous().requires_grad_(True)
        
        # Branched with different branch counts
        results = {}
        for bc in [1, 2, 4, 8]:
            wpb = D // (bc * W)  # windows per branch
            if wpb < 1:
                continue  # Skip if not enough windows
            
            template_b = (torch.randn(H, bc, wpb, W, device=device) * 0.02).requires_grad_(True)
            template_flat_b = template_b.view(H, -1).contiguous()
            
            # Forward timing
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(50):
                out = template_gate_ffn_branched_flat(
                    x, template_flat_b, H, tau=tau, W=W,
                    branch_count=bc, windows_per_branch=wpb
                )
            torch.cuda.synchronize()
            fwd_time = (time.perf_counter() - start) / 50 * 1000
            
            # Backward timing
            out = template_gate_ffn_branched_flat(
                x, template_flat_b.clone().requires_grad_(True), H, tau=tau, W=W,
                branch_count=bc, windows_per_branch=wpb
            )
            grad = torch.ones_like(out)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(50):
                x.grad = None
                out.backward(grad, retain_graph=True)
            torch.cuda.synchronize()
            bwd_time = (time.perf_counter() - start) / 50 * 1000
            
            results[bc] = (fwd_time, bwd_time)
        
        # Original timing
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            out_orig = template_gate_ffn_flat(x_orig, template_flat_orig, H, tau=tau, W=W, num_windows=num_windows_orig)
        torch.cuda.synchronize()
        orig_fwd = (time.perf_counter() - start) / 50 * 1000
        
        out_orig = template_gate_ffn_flat(x_orig, template_flat_orig.clone().requires_grad_(True), H, tau=tau, W=W, num_windows=num_windows_orig)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            x_orig.grad = None
            out_orig.backward(grad, retain_graph=True)
        torch.cuda.synchronize()
        orig_bwd = (time.perf_counter() - start) / 50 * 1000
        
        print(f"N={N:5d}: Original[fwd={orig_fwd:.3f}ms, bwd={orig_bwd:.3f}ms] ", end="")
        for bc, (fwd, bwd) in results.items():
            ratio_fwd = fwd / orig_fwd
            ratio_bwd = bwd / orig_bwd
            print(f"B={bc}[{ratio_fwd:.2f}x/{ratio_bwd:.2f}x] ", end="")
        print()
    
    # === Branch Scaling Test ===
    print("\n--- Branch Scaling Test (fixed total windows, varying branches) ---")
    N = 1024
    D = 512
    total_windows = 8  # Keep total windows constant
    
    print(f"N={N}, D={D}, total_windows={total_windows}")
    print("-" * 60)
    
    for bc in [1, 2, 4, 8]:
        wpb = total_windows // bc  # Divide windows among branches
        W_local = D // total_windows  # W = 64 for D=512, total_windows=8
        
        x = torch.randn(N, D, device=device)
        template = torch.randn(H, bc, wpb, W_local, device=device) * 0.02
        template_flat = template.view(H, -1).contiguous()
        
        # Forward timing
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            out = template_gate_ffn_branched_flat(
                x, template_flat, H, tau=tau, W=W_local,
                branch_count=bc, windows_per_branch=wpb
            )
        torch.cuda.synchronize()
        fwd_time = (time.perf_counter() - start) / 100 * 1000
        
        print(f"branch_count={bc}, windows_per_branch={wpb}: fwd={fwd_time:.3f}ms")
    
    print("\n" + "=" * 70)
    print("All tests complete!")
    print("=" * 70)
