"""
Quaternion Rotation Primitives for the Spinformer
=================================================

Each 4D chunk of the residual stream is treated as a quaternion.
Learned rotation quaternions (axis + angle) rotate each chunk,
acting as a geometric pre-processing filter before layers.

Quaternion multiplication: q * p where q = (w, x, y, z)
  (q*p).w = qw*pw - qx*px - qy*py - qz*pz
  (q*p).x = qw*px + qx*pw + qy*pz - qz*py
  (q*p).y = qw*py - qx*pz + qy*pw + qz*px
  (q*p).z = qw*pz + qx*py - qy*px + qz*pw

Rotation of v by unit quaternion q: v' = q * v * conj(q)
"""

import torch
import torch.nn as nn
import math

# Try to import Triton kernel, fallback to PyTorch
_HAS_TRITON = False
try:
    from .triton_kernels import triton_quaternion_rotate
    _HAS_TRITON = True
except ImportError:
    pass


def quaternion_multiply(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product of two quaternion tensors.
    q, p: (..., 4) where dim -1 is (w, x, y, z)
    Returns: (..., 4)
    """
    qw, qx, qy, qz = q.unbind(-1)
    pw, px, py, pz = p.unbind(-1)
    
    ow = qw*pw - qx*px - qy*py - qz*pz
    ox = qw*px + qx*pw + qy*pz - qz*py
    oy = qw*py - qx*pz + qy*pw + qz*px
    oz = qw*pz + qx*py - qy*px + qz*pw
    
    return torch.stack([ow, ox, oy, oz], dim=-1)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate: (w, -x, -y, -z)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def axis_angle_to_quaternion(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis (unit vector, shape (..., 3)) and angle (shape (...,)) to unit quaternion (..., 4).
    q = (cos(θ/2), sin(θ/2) * axis)
    """
    half = angle * 0.5
    s = torch.sin(half)
    c = torch.cos(half)
    # axis: (..., 3), s: (...)
    return torch.cat([c.unsqueeze(-1), s.unsqueeze(-1) * axis], dim=-1)


def quaternion_rotate(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Rotate 4D vectors v by unit quaternions q via: v' = q * v * conj(q)
    v: (..., 4)  — treated as pure quaternions (w=v[0], x=v[1], y=v[2], z=v[3])
    q: (..., 4)  — unit quaternions
    Returns: (..., 4)
    """
    q_conj = quaternion_conjugate(q)
    return quaternion_multiply(quaternion_multiply(q, v), q_conj)


class QuaternionRotationLayer(nn.Module):
    """
    Quaternion rotation pre-filter for the Spinformer.
    
    Given input x of shape [B, S, D], chunks it into D/4 quaternion groups,
    computes per-chunk rotation angles via a learned linear projection,
    and rotates each chunk around its learned axis.
    
    Architecture:
    1. x_normed = RMSNorm(x)  [done externally or optionally here]
    2. angles = Linear(x_normed, D -> D/4)  [one angle per 4D chunk]
    3. For each chunk i: q_i = axis_angle_to_quat(learned_axis_i, angle_i)
    4. x_rotated[chunk_i] = q_i * x[chunk_i] * conj(q_i)
    
    The axes are NOT input-dependent (learned parameters).
    The angles ARE input-dependent (projected from the input).
    """
    def __init__(self, dim: int, use_triton: bool = False):
        super().__init__()
        assert dim % 4 == 0, f"dim must be divisible by 4, got {dim}"
        self.dim = dim
        self.n_chunks = dim // 4
        
        # Use PyTorch ops by default (torch.compile fuses them better than custom Triton + graph breaks)
        # Set use_triton=True only for non-compiled inference
        self._use_triton = use_triton and _HAS_TRITON
        
        # Learned rotation axes (one 3D unit vector per chunk)
        # Parameterized as unconstrained 3D vector, normalized at forward time
        self.raw_axes = nn.Parameter(torch.randn(self.n_chunks, 3) * 0.01)
        
        # Angle projection: D -> D/4 (one scalar angle per 4D chunk)
        self.angle_proj = nn.Linear(dim, self.n_chunks, bias=False)
        nn.init.zeros_(self.angle_proj.weight)  # Start near identity
        
    def get_unit_axes(self) -> torch.Tensor:
        """Normalize raw axes to unit vectors. Shape: (n_chunks, 3)"""
        return torch.nn.functional.normalize(self.raw_axes, dim=-1)
    
    def _rotate_triton(self, x_chunks: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return triton_quaternion_rotate(x_chunks, q)
    
    def _rotate_pytorch(self, x_chunks: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return quaternion_rotate(x_chunks, q)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, D]
        Returns: [B, S, D] rotated
        """
        B, S, D = x.shape
        
        # 1. Compute rotation angles from input (input-dependent)
        angles = self.angle_proj(x)  # [B, S, n_chunks]
        
        # 2. Get unit axes (not input-dependent)
        axes = self.get_unit_axes()  # [n_chunks, 3]
        
        # 3. Build rotation quaternions
        axes_expanded = axes.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
        q = axis_angle_to_quaternion(axes_expanded, angles)  # [B, S, n_chunks, 4]
        
        # 4. Reshape x into chunks: [B, S, n_chunks, 4]
        x_chunks = x.view(B, S, self.n_chunks, 4)
        
        # 5. Apply rotation (backend chosen at construction, no runtime branch)
        if self._use_triton:
            x_rotated = self._rotate_triton(x_chunks, q)
        else:
            x_rotated = self._rotate_pytorch(x_chunks, q)
        
        # 6. Reshape back: [B, S, D]
        return x_rotated.view(B, S, D)
