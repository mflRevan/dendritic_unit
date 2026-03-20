"""
Triton Kernels for Quaternion Rotation
=======================================

Fused forward + backward kernels for quaternion rotation of 4D chunks.
Avoids materializing intermediate quaternion products — everything in registers.

The rotation v' = q * v * q* expands to (for unit quaternion q = (w, x, y, z)):

v'_0 = v0*(w²+x²-y²-z²) + v1*2*(xy-wz)      + v2*2*(xz+wy)      + v3*2*(wx-yz) -- wait, 
Actually for 4D quaternion-on-quaternion rotation, the formula is:
v' = q * v * conj(q)  (Hamilton product chain)

We compute this directly via two sequential Hamilton products,
keeping everything in registers per-element.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _quat_hamilton_product(
    aw, ax, ay, az,
    bw, bx, by, bz,
):
    """Inline Hamilton product: returns (w, x, y, z) of a*b"""
    ow = aw*bw - ax*bx - ay*by - az*bz
    ox = aw*bx + ax*bw + ay*bz - az*by
    oy = aw*by - ax*bz + ay*bw + az*bx
    oz = aw*bz + ax*by - ay*bx + az*bw
    return ow, ox, oy, oz


@triton.jit
def quaternion_rotate_fwd_kernel(
    # Pointers
    X_ptr,          # Input: [N, 4] flattened from [B, S, n_chunks, 4]
    Q_ptr,          # Quaternions: [N, 4] 
    Out_ptr,        # Output: [N, 4]
    # Dims
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Forward: out = q * x * conj(q), element-wise over N quaternion pairs."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # Load x (as quaternion)
    xw = tl.load(X_ptr + offs * 4 + 0, mask=mask, other=0.0)
    xx = tl.load(X_ptr + offs * 4 + 1, mask=mask, other=0.0)
    xy = tl.load(X_ptr + offs * 4 + 2, mask=mask, other=0.0)
    xz = tl.load(X_ptr + offs * 4 + 3, mask=mask, other=0.0)
    
    # Load q (rotation quaternion)
    qw = tl.load(Q_ptr + offs * 4 + 0, mask=mask, other=1.0)
    qx = tl.load(Q_ptr + offs * 4 + 1, mask=mask, other=0.0)
    qy = tl.load(Q_ptr + offs * 4 + 2, mask=mask, other=0.0)
    qz = tl.load(Q_ptr + offs * 4 + 3, mask=mask, other=0.0)
    
    # Step 1: tmp = q * x
    tw, tx, ty, tz = _quat_hamilton_product(qw, qx, qy, qz, xw, xx, xy, xz)
    
    # Step 2: out = tmp * conj(q) = tmp * (qw, -qx, -qy, -qz)
    ow, ox, oy, oz = _quat_hamilton_product(tw, tx, ty, tz, qw, -qx, -qy, -qz)
    
    # Store
    tl.store(Out_ptr + offs * 4 + 0, ow, mask=mask)
    tl.store(Out_ptr + offs * 4 + 1, ox, mask=mask)
    tl.store(Out_ptr + offs * 4 + 2, oy, mask=mask)
    tl.store(Out_ptr + offs * 4 + 3, oz, mask=mask)


@triton.jit
def quaternion_rotate_bwd_kernel(
    # Pointers
    Grad_out_ptr,   # dL/d(out): [N, 4]
    X_ptr,          # Original input: [N, 4]
    Q_ptr,          # Rotation quaternions: [N, 4]
    Grad_x_ptr,     # dL/dx: [N, 4]
    Grad_q_ptr,     # dL/dq: [N, 4]
    # Dims
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward pass for v' = q * v * q*.
    
    Given grad_out = dL/dv':
    - dL/dv = q* * grad_out * q  (inverse rotation of gradient)
    - dL/dq = 2 * (grad_out * q * v* - v * q* * grad_out*) ... 
    
    Actually, the cleanest approach:
    dL/dx: since rotation is linear in x, and rotation by q is orthogonal:
        dL/dx = q* * dL/dout * q  (apply inverse rotation to gradient)
    
    dL/dq: we use the identity that for unit q:
        d(q*v*q*)/dq involves the commutator.
        Concretely: dL/dq = 2 * (dL/dout * conj(q) * conj(v) + conj(conj(v) * conj(q) * conj(dL/dout)))
        
    But simpler: since out_i = q * v * q*, and using the product rule:
        d(out)/dq = [expanding q*v*q*] 
        
    Let's use the fact that:
        out = q * (v * q*)  => product rule:  d(out)/dq applied to dq gives:
            dq * (v * q*) + q * (v * dq*)
            = dq * (v * q*) + q * v * conj(dq)
    
    For the gradient wrt q, we need:
        dL/dq_w, dL/dq_x, dL/dq_y, dL/dq_z
    
    Using: dL/dq = 2 * real_part_extract(grad_out * q_conj * x_conj)  ... this is standard.
    
    Actually the simplest correct derivation for unit quaternion rotation gradient:
    
    Let g = dL/d(out), a = q * v * q*  (forward)
    
    dL/dv (treating v as the variable): 
        v' = q * v * q*  is a linear map in v, and the adjoint of 
        "left-multiply by q, right-multiply by q*" is "left-multiply by q*, right-multiply by q"
        So: dL/dv = q* * g * q
    
    dL/dq (treating q as the variable, with unit constraint):
        Using product rule on q * v * q*:
        d(out) = dq * v * q* + q * v * d(q*)
               = dq * v * q* + q * v * conj(dq)    [since d(conj(q)) = conj(dq)]
        
        dL = <g, d(out)> = <g, dq * v * q*> + <g, q * v * conj(dq)>
        
        <g, dq * v * q*> = <g * q * v*, dq>   (using <a*b*c, d> = <a, d*conj(c)*conj(b)>... )
        
        Actually let's just let the gradient wrt q be:
        dL/dq = g * conj(v * q*) + conj(v * q*) ... no.
        
        The standard result for Hamilton product gradient:
        If c = a * b, then:
            dc/da . da = da * b     =>  dL/da = dL/dc * conj(b)  ... NO. 
        
        Hamilton product is NOT commutative, so we need to be careful.
        For c = a * b:
            dL/da: since c = a*b, varying a: dc = da * b
                <dL/dc, dc> = <dL/dc, da * b>
                This equals the real part of conj(dL/dc) * da * b ... 
                Actually in 4D: dL/da_i = sum_j (dL/dc_j * d(c_j)/d(a_i))
                The Jacobian of the Hamilton product wrt the left factor is a 4x4 matrix
                that depends on b.
            
        Let me use the Jacobian approach directly.
        For c = a * b where a=(a0,a1,a2,a3), b=(b0,b1,b2,b3):
        dc/da is the 4x4 matrix:
            [[b0, -b1, -b2, -b3],
             [b1,  b0,  b3, -b2],
             [b2, -b3,  b0,  b1],
             [b3,  b2, -b1,  b0]]
        
        So dL/da = (dc/da)^T * dL/dc = R_b^T * g  where R_b is the right-multiply-by-b matrix.
        
        For c = a * b:
            dL/da = g * conj(b)   ... this is the formula! Because R_b^T for unit b equals L_{conj(b)}.
            dL/db = conj(a) * g   ... and L_a^T for unit a equals R_{conj(a)}.
        
        Wait no. Let me verify:
        c = a * b
        dL/da should satisfy: <dL/da, da> = <g, da * b> for all da
        If we set dL/da such that the inner product works:
        In quaternion algebra, <p, q> = p0*q0 + p1*q1 + p2*q2 + p3*q3 (real dot product)
        <g, da * b> = Re(conj(g) * (da * b))  ... hmm this gets complicated.
        
        Let me just use the Jacobian directly in the kernel.
        
    For practical purposes, I'll implement the backward via:
    
    Forward: out = q * x * q*
    
    Split into: t = q * x,  out = t * q*
    
    Backward of out = t * q*:
        dL/dt = dL/dout * conj(q*)  = g * q    (right-mul by conj of right factor)
        For dL/dq*: need conj(t) * g = conj(q*x) * g
        But we need dL/dq, not dL/dq*.
        Since q* maps to q via conjugation: dL/dq = conj(dL/dq*)
        
    Backward of t = q * x:
        dL/dq += dL/dt * conj(x)   (right-mul by conj of right factor)
        dL/dx = conj(q) * dL/dt    (left-mul by conj of left factor)
    
    So overall:
        step1: g_t = g * q                       (dL/dt)
        step2: dL/dx = conj(q) * g_t = q* * g * q   (inverse rotation - correct!)
        
        For dL/dq:
        From out = t * q*:  contribution1 = conj(t) * g  ... but this gives dL/d(q*)
            so contribution1_to_q = conj(conj(t) * g)  -- conjugate the whole thing
            Wait no. Let me think again.
            
        out = t * q*
        d(out) = dt * q* + t * d(q*)
        <g, t * d(q*)> = ... we need dL/d(q*) and then chain to dL/dq.
        
        d(q*) = conj(dq)  (conjugation is linear and self-inverse)
        So <g, t * conj(dq)>
        
        Hmm, let me just be pragmatic and use the Jacobian matrices.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # Load grad_out
    gw = tl.load(Grad_out_ptr + offs * 4 + 0, mask=mask, other=0.0)
    gx = tl.load(Grad_out_ptr + offs * 4 + 1, mask=mask, other=0.0)
    gy = tl.load(Grad_out_ptr + offs * 4 + 2, mask=mask, other=0.0)
    gz = tl.load(Grad_out_ptr + offs * 4 + 3, mask=mask, other=0.0)
    
    # Load x
    xw = tl.load(X_ptr + offs * 4 + 0, mask=mask, other=0.0)
    xx = tl.load(X_ptr + offs * 4 + 1, mask=mask, other=0.0)
    xy = tl.load(X_ptr + offs * 4 + 2, mask=mask, other=0.0)
    xz = tl.load(X_ptr + offs * 4 + 3, mask=mask, other=0.0)
    
    # Load q
    qw = tl.load(Q_ptr + offs * 4 + 0, mask=mask, other=1.0)
    qx = tl.load(Q_ptr + offs * 4 + 1, mask=mask, other=0.0)
    qy = tl.load(Q_ptr + offs * 4 + 2, mask=mask, other=0.0)
    qz = tl.load(Q_ptr + offs * 4 + 3, mask=mask, other=0.0)
    
    # --- dL/dx = q* * g * q  (inverse rotation of gradient) ---
    # Step 1: tmp = q* * g
    tw, tx, ty, tz = _quat_hamilton_product(qw, -qx, -qy, -qz, gw, gx, gy, gz)
    # Step 2: dx = tmp * q
    dxw, dxx, dxy, dxz = _quat_hamilton_product(tw, tx, ty, tz, qw, qx, qy, qz)
    
    tl.store(Grad_x_ptr + offs * 4 + 0, dxw, mask=mask)
    tl.store(Grad_x_ptr + offs * 4 + 1, dxx, mask=mask)
    tl.store(Grad_x_ptr + offs * 4 + 2, dxy, mask=mask)
    tl.store(Grad_x_ptr + offs * 4 + 3, dxz, mask=mask)
    
    # --- dL/dq ---
    # Forward: out = q * x * q*
    # Split: t = q * x, out = t * q*
    # 
    # For c = a * b, the Jacobian dc/da (treating quaternions as R^4 vectors):
    #   dc/da = R_b  (right-multiplication matrix for b)
    #   R_b = [[b0, -b1, -b2, -b3],
    #          [b1,  b0,  b3, -b2],
    #          [b2, -b3,  b0,  b1],
    #          [b3,  b2, -b1,  b0]]
    #   dL/da = R_b^T @ g
    #
    # For t = q * x:  dL/dq_from_t = R_x^T @ dL/dt
    # For out = t * q*:  dL/dt = R_{q*}^T @ g   and   dL/d(q*) = L_t^T @ g
    #   where L_a = [[a0, -a1, -a2, -a3],
    #                [a1,  a0, -a3,  a2],
    #                [a2,  a3,  a0, -a1],
    #                [a3, -a2,  a1,  a0]]
    # 
    # dL/dq from the q* path: dL/dq = J_{q->q*}^T @ dL/d(q*) = conj_jacobian^T @ L_t^T @ g
    # Since conj_jacobian = diag(1, -1, -1, -1), its transpose is itself.
    # So dL/dq_from_qstar = diag(1,-1,-1,-1) @ L_t^T @ g
    
    # First compute t = q * x
    t0, t1, t2, t3 = _quat_hamilton_product(qw, qx, qy, qz, xw, xx, xy, xz)
    
    # Compute dL/dt = R_{q*}^T @ g
    # R_{q*} uses (qw, -qx, -qy, -qz)
    # R_b^T: row i of R_b becomes column i => 
    # R_b^T @ g:
    #   [b0*g0 + b1*g1 + b2*g2 + b3*g3,
    #    -b1*g0 + b0*g1 - b3*g2 + b2*g3,
    #    -b2*g0 + b3*g1 + b0*g2 - b1*g3,
    #    -b3*g0 - b2*g1 + b1*g2 + b0*g3]
    # with b = q* = (qw, -qx, -qy, -qz)
    cqx = -qx  # components of q*
    cqy = -qy
    cqz = -qz
    
    dtw = qw*gw + cqx*gx + cqy*gy + cqz*gz
    dtx = -cqx*gw + qw*gx - cqz*gy + cqy*gz
    dty = -cqy*gw + cqz*gx + qw*gy - cqx*gz
    dtz = -cqz*gw - cqy*gx + cqx*gy + qw*gz
    
    # Now dL/dq contribution 1: from t = q * x, dL/dq = R_x^T @ dL/dt
    # R_x^T @ dt:
    dq1w = xw*dtw + xx*dtx + xy*dty + xz*dtz
    dq1x = -xx*dtw + xw*dtx - xz*dty + xy*dtz
    dq1y = -xy*dtw + xz*dtx + xw*dty - xx*dtz
    dq1z = -xz*dtw - xy*dtx + xx*dty + xw*dtz
    
    # Now dL/dq contribution 2: from out = t * q*, dL/d(q*) = L_t^T @ g
    # L_a^T @ g:
    #   [a0*g0 + a1*g1 + a2*g2 + a3*g3,
    #    -a1*g0 + a0*g1 + a3*g2 - a2*g3,
    #    -a2*g0 - a3*g1 + a0*g2 + a1*g3,
    #    -a3*g0 + a2*g1 - a1*g2 + a0*g3]
    dqsw = t0*gw + t1*gx + t2*gy + t3*gz
    dqsx = -t1*gw + t0*gx + t3*gy - t2*gz
    dqsy = -t2*gw - t3*gx + t0*gy + t1*gz
    dqsz = -t3*gw + t2*gx - t1*gy + t0*gz
    
    # Apply conjugation Jacobian: diag(1, -1, -1, -1) to get dL/dq from dL/d(q*)
    dq2w = dqsw
    dq2x = -dqsx
    dq2y = -dqsy
    dq2z = -dqsz
    
    # Total dL/dq
    dqw_total = dq1w + dq2w
    dqx_total = dq1x + dq2x
    dqy_total = dq1y + dq2y
    dqz_total = dq1z + dq2z
    
    tl.store(Grad_q_ptr + offs * 4 + 0, dqw_total, mask=mask)
    tl.store(Grad_q_ptr + offs * 4 + 1, dqx_total, mask=mask)
    tl.store(Grad_q_ptr + offs * 4 + 2, dqy_total, mask=mask)
    tl.store(Grad_q_ptr + offs * 4 + 3, dqz_total, mask=mask)


class TritonQuaternionRotate(torch.autograd.Function):
    """Custom autograd function using Triton kernels for quaternion rotation."""
    
    BLOCK_SIZE = 1024
    
    @staticmethod
    def forward(ctx, x_chunks: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        x_chunks: [B, S, n_chunks, 4] — input quaternion chunks
        q: [B, S, n_chunks, 4] — rotation quaternions (unit)
        Returns: [B, S, n_chunks, 4] — rotated
        """
        shape = x_chunks.shape
        N = x_chunks.numel() // 4  # total number of quaternion elements
        
        x_flat = x_chunks.contiguous().view(-1, 4)
        q_flat = q.contiguous().view(-1, 4)
        out_flat = torch.empty_like(x_flat)
        
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        quaternion_rotate_fwd_kernel[grid](
            x_flat, q_flat, out_flat,
            N=N,
            BLOCK_SIZE=TritonQuaternionRotate.BLOCK_SIZE,
        )
        
        ctx.save_for_backward(x_flat, q_flat)
        ctx.shape = shape
        return out_flat.view(shape)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_flat, q_flat = ctx.saved_tensors
        shape = ctx.shape
        N = x_flat.shape[0]
        
        grad_flat = grad_output.contiguous().view(-1, 4)
        grad_x = torch.empty_like(x_flat)
        grad_q = torch.empty_like(q_flat)
        
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        quaternion_rotate_bwd_kernel[grid](
            grad_flat, x_flat, q_flat,
            grad_x, grad_q,
            N=N,
            BLOCK_SIZE=TritonQuaternionRotate.BLOCK_SIZE,
        )
        
        return grad_x.view(shape), grad_q.view(shape)


def triton_quaternion_rotate(x_chunks: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Functional interface for Triton quaternion rotation."""
    return TritonQuaternionRotate.apply(x_chunks, q)
