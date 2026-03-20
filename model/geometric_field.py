"""
Geometric Weight-Field Modulation
==================================
Generate linear operator weights from learned latent geometric objects.

Core idea: Instead of storing W as an unconstrained dense matrix, parameterize
it (or modulate it) via a set of learnable coordinates that are rotated, scaled,
and decoded into weight-space. Context can condition the rotation/scale/pivot,
making the operator a function of the input.

Primary target: attention output projection (W_O).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal


# ---------- quaternion helpers (pure pytorch, compile-friendly) ----------

def _quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Build unit quaternion [w, x, y, z] from 3-axis + angle (radians).
    axis: (..., 3)  angle: (...)  -> (..., 4)"""
    axis = F.normalize(axis, dim=-1)
    half = angle.unsqueeze(-1) * 0.5          # (..., 1)
    w = half.cos()                             # (..., 1)
    xyz = axis * half.sin()                    # (..., 3)
    return torch.cat([w, xyz], dim=-1)         # (..., 4)


def _quat_rotate_points(q: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Rotate 3-D points by unit quaternion.
    q: (..., 4)  pts: (..., N, 3) -> (..., N, 3)
    Uses the Rodrigues-like formula: v' = v + 2*w*(t × v) + 2*(t × (t × v))
    where t = xyz part of quaternion."""
    t = q[..., 1:4]    # (..., 3)
    w = q[..., 0:1]    # (..., 1)

    def _cross(a, b):
        return torch.stack([
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ], dim=-1)

    # Broadcast t and w to match pts: (..., N, 3) and (..., N, 1)
    t = t.unsqueeze(-2)   # (..., 1, 3)
    w = w.unsqueeze(-1)   # (..., 1, 1)
    txv = _cross(t.expand_as(pts), pts)
    return pts + 2.0 * w * txv + 2.0 * _cross(t.expand_as(txv), txv)


# ---------- Cayley rotation helpers (arbitrary dim, compile-friendly) ----------

def _build_skew_symmetric(params: torch.Tensor, dim: int) -> torch.Tensor:
    """Build skew-symmetric matrix from upper-triangular parameters.
    params: (..., D*(D-1)/2) -> (..., D, D)"""
    *batch_shape, _ = params.shape
    idx = torch.triu_indices(dim, dim, offset=1, device=params.device)
    A = params.new_zeros(*batch_shape, dim, dim)
    A[..., idx[0], idx[1]] = params
    return A - A.transpose(-1, -2)


def _cayley_rotate_points(skew_params: torch.Tensor, angle: torch.Tensor,
                          pts: torch.Tensor, dim: int) -> torch.Tensor:
    """Rotate points via Cayley map: R = (I+A)^{-1}(I-A) where A is skew-symmetric.
    skew_params: (H, D*(D-1)/2), angle: (H,)|(B,H)|(B,S,H), pts: (H, N, D)
    Returns: rotated (..., H, N, D)"""
    I = torch.eye(dim, device=pts.device, dtype=pts.dtype)
    if angle.dim() == 1:
        A = _build_skew_symmetric(skew_params, dim) * angle[:, None, None]
        R = torch.linalg.solve(I + A, (I - A).expand_as(A))  # (H, D, D)
        return torch.bmm(pts, R.transpose(-1, -2))
    elif angle.dim() == 2:
        B_sz = angle.shape[0]
        sp = skew_params.unsqueeze(0).expand(B_sz, -1, -1)
        A = _build_skew_symmetric(sp, dim) * angle[:, :, None, None]
        R = torch.linalg.solve(I + A, (I - A).expand_as(A))
        pts_exp = pts.unsqueeze(0).expand(B_sz, -1, -1, -1)
        return torch.einsum('bhnd,bhde->bhne', pts_exp, R)
    else:
        B_sz, S, H = angle.shape
        sp = skew_params.unsqueeze(0).unsqueeze(0).expand(B_sz, S, -1, -1)
        A = _build_skew_symmetric(sp, dim) * angle[..., None, None]
        R = torch.linalg.solve(I + A, (I - A).expand_as(A))
        pts_exp = pts.unsqueeze(0).unsqueeze(0).expand(B_sz, S, -1, -1, -1)
        return torch.einsum('bshnd,bshde->bshne', pts_exp, R)


# ---------- Linear perturbation helpers ----------

def _linear_perturb_points(direction: torch.Tensor, angle: torch.Tensor,
                           pts: torch.Tensor) -> torch.Tensor:
    """Additive perturbation: pts + angle * normalize(direction).
    direction: (H, D), angle: (H,)|(B,H)|(B,S,H), pts: (H, N, D)
    Returns: perturbed (..., H, N, D)"""
    d = F.normalize(direction, dim=-1)
    if angle.dim() == 1:
        delta = (angle[:, None] * d).unsqueeze(1)             # (H, 1, D)
        return pts + delta
    elif angle.dim() == 2:
        delta = (angle.unsqueeze(-1) * d.unsqueeze(0)).unsqueeze(2)  # (B, H, 1, D)
        return pts.unsqueeze(0) + delta
    else:
        delta = (angle.unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)).unsqueeze(3)  # (B, S, H, 1, D)
        return pts.unsqueeze(0).unsqueeze(0) + delta


# ---------- Coordinate decoders ----------

class LinearDecoder(nn.Module):
    """Decode flattened transformed coordinates into a weight matrix (or low-rank factors)."""
    def __init__(self, num_coords: int, coord_dim: int, out_features: int, in_features: int,
                 rank: int = 0):
        super().__init__()
        flat_dim = num_coords * coord_dim
        self.out_features = out_features
        self.in_features = in_features
        self.rank = rank

        if rank > 0:
            # Low-rank: decode coords -> (rank,), then outer-product via U, V
            self.coord_to_rank = nn.Linear(flat_dim, rank, bias=False)
            self.U = nn.Linear(rank, out_features, bias=False)
            self.V = nn.Linear(rank, in_features, bias=False)
        else:
            # Full decode: coords -> flat weight
            self.proj = nn.Linear(flat_dim, out_features * in_features, bias=False)

    def forward(self, coords_flat: torch.Tensor) -> torch.Tensor:
        """coords_flat: (..., num_coords * coord_dim) -> (..., out, in)."""
        if self.rank > 0:
            r = self.coord_to_rank(coords_flat)    # (..., rank)
            u = self.U.weight                      # (out, rank)
            v = self.V.weight                      # (in, rank)
            # W = U diag(r) V^T  i.e. sum_k u_k * r_k * v_k^T
            # Efficient: (..., out, rank) * (..., rank, in) with r scaling
            if coords_flat.dim() == 1:
                return (u * r.unsqueeze(0)) @ v.t()   # (out, in)
            # batched
            u_r = u.unsqueeze(0) * r.unsqueeze(-2)    # (..., out, rank)
            return u_r @ v.t().unsqueeze(0)            # (..., out, in)
        else:
            w_flat = self.proj(coords_flat)
            shape = coords_flat.shape[:-1] + (self.out_features, self.in_features)
            return w_flat.view(shape)


# ---------- Core geometric weight-field module ----------

class GeometricWeightField(nn.Module):
    """
    Generates or modulates a weight matrix from learned latent coordinates
    transformed by quaternion rotation, optional scaling, and optional pivot.

    Modes
    -----
    - "replace"    : W_eff = G(c; Θ)
    - "residual"   : W_eff = W_base + λ · G(c; Θ)          [default]
    - "factorized" : W_eff = W_base + U · G(c; Θ) · V^T    [low-rank]

    Conditioning
    ------------
    - "static"          : learned angle per layer (no input dependence)
    - "seq_conditioned" : angle from mean-pooled hidden state
    - "token_conditioned" : per-token angle (expensive, deferred)

    Geometry components
    -------------------
    - rotation (quaternion, always on)
    - scale   (per-axis, optional)
    - pivot   (centroid by default, optional learned offset)
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        *,
        num_coords: int = 32,
        coord_dim: int = 3,
        mode: Literal["replace", "residual", "factorized"] = "residual",
        conditioning: Literal["static", "seq_conditioned", "token_conditioned"] = "static",
        use_scale: bool = False,
        use_pivot_offset: bool = False,
        rank: int = 0,          # 0 = full decode, >0 = low-rank
        lam_init: float = 0.1,  # initial λ for residual / factorized
        num_heads: int = 1,     # >1 for per-head transforms
        cond_dim: int = 128,    # hidden dim of conditioning input
        rotation_type: str = "quaternion",  # quaternion, cayley, linear
    ):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.num_coords = num_coords
        self.coord_dim = coord_dim
        self.mode = mode
        self.conditioning = conditioning
        self.num_heads = num_heads
        self.rotation_type = rotation_type

        # ---- learnable latent coordinates ----
        # Shape: (num_heads, num_coords, coord_dim)  or (1, N, 3) if shared
        self.coords = nn.Parameter(
            torch.randn(num_heads, num_coords, coord_dim) * 0.1
        )

        # ---- rotation parameters (type-dependent) ----
        if rotation_type == "quaternion":
            assert coord_dim == 3, "Quaternion rotation requires coord_dim=3"
            self.axis = nn.Parameter(torch.randn(num_heads, coord_dim))
        elif rotation_type == "cayley":
            n_skew = coord_dim * (coord_dim - 1) // 2
            self.skew_params = nn.Parameter(torch.randn(num_heads, n_skew) * 0.1)
        elif rotation_type == "linear":
            self.perturb_dir = nn.Parameter(torch.randn(num_heads, coord_dim))
        else:
            raise ValueError(f"Unknown rotation_type: {rotation_type}")

        # ---- angle ----
        if conditioning == "static":
            self.angle = nn.Parameter(torch.zeros(num_heads))
        else:  # seq_conditioned or token_conditioned
            self.angle_proj = nn.Linear(cond_dim, num_heads, bias=True)
            nn.init.zeros_(self.angle_proj.weight)
            nn.init.zeros_(self.angle_proj.bias)

        # ---- optional scale (per-axis, per head) ----
        self.use_scale = use_scale
        if use_scale:
            self.scale_logit = nn.Parameter(torch.zeros(num_heads, coord_dim))

        # ---- optional pivot offset ----
        self.use_pivot_offset = use_pivot_offset
        if use_pivot_offset:
            self.pivot_offset = nn.Parameter(torch.zeros(num_heads, coord_dim))

        # ---- decoder ----
        if num_heads > 1:
            # Per-head: each head decodes a (out_per_head, in_features) block
            assert out_features % num_heads == 0
            out_per_head = out_features // num_heads
            self.decoder = LinearDecoder(num_coords, coord_dim, out_per_head, in_features,
                                         rank=rank)
        else:
            self.decoder = LinearDecoder(num_coords, coord_dim, out_features, in_features,
                                         rank=rank)

        # ---- blend scale λ (for residual / factorized) ----
        if mode in ("residual", "factorized"):
            self.lam = nn.Parameter(torch.tensor(lam_init))

        # ---- base weight (for residual mode) ----
        if mode == "residual":
            self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
            self.base_bias = nn.Parameter(torch.zeros(out_features))
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        elif mode == "replace":
            self.register_parameter("base_weight", None)
            self.base_bias = nn.Parameter(torch.zeros(out_features))
        # factorized mode expects external base weight

    def _get_angle(self, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return rotation angle(s).  Shape: (num_heads,) or (B, num_heads)."""
        if self.conditioning == "static":
            return self.angle                          # (H,)
        else:
            assert context is not None
            return self.angle_proj(context)            # (B, H)

    def _quaternion_rotate(self, centered: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Quaternion rotation (3D only). centered: (H, N, 3), angle: various dims."""
        if angle.dim() == 1:
            q = _quat_from_axis_angle(self.axis, angle)
            return _quat_rotate_points(q, centered)
        elif angle.dim() == 2:
            axis = self.axis.unsqueeze(0).expand(angle.shape[0], -1, -1)
            q = _quat_from_axis_angle(axis, angle)
            centered_exp = centered.unsqueeze(0).expand(angle.shape[0], -1, -1, -1)
            return _quat_rotate_points(q, centered_exp)
        else:
            B, S, H = angle.shape
            axis = self.axis.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
            q = _quat_from_axis_angle(axis, angle)
            centered_exp = centered.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1, -1)
            return _quat_rotate_points(q, centered_exp)

    def _transform_coords(self, angle: torch.Tensor) -> torch.Tensor:
        """Apply geometric transform to coordinates.
        angle: (H,) for static, (B, H) for seq_conditioned, (B, S, H) for token_conditioned
        Returns: transformed coords, (..., H, N, D)
        """
        # Pivot = centroid of coordinates + optional offset
        pivot = self.coords.mean(dim=1, keepdim=True)  # (H, 1, D)
        if self.use_pivot_offset:
            pivot = pivot + self.pivot_offset.unsqueeze(1)

        # Center around pivot
        centered = self.coords - pivot                  # (H, N, D)

        # Apply rotation/perturbation based on type
        if self.rotation_type == "quaternion":
            rotated = self._quaternion_rotate(centered, angle)
        elif self.rotation_type == "cayley":
            rotated = _cayley_rotate_points(self.skew_params, angle, centered, self.coord_dim)
        elif self.rotation_type == "linear":
            rotated = _linear_perturb_points(self.perturb_dir, angle, centered)
        else:
            raise ValueError(f"Unknown rotation_type: {self.rotation_type}")

        # Optional scale
        if self.use_scale:
            s = F.softplus(self.scale_logit)  # (H, D)
            if rotated.dim() == 3:
                rotated = rotated * s.unsqueeze(1)       # (H, N, D)
            elif rotated.dim() == 4:
                rotated = rotated * s.unsqueeze(0).unsqueeze(2)  # (B, H, N, D)
            else:
                rotated = rotated * s.unsqueeze(0).unsqueeze(0).unsqueeze(3)  # (B, S, H, N, D)

        # Uncenter
        if rotated.dim() == 3:
            return rotated + pivot                        # (H, N, D)
        elif rotated.dim() == 4:
            return rotated + pivot.unsqueeze(0)           # (B, H, N, D)
        else:
            return rotated + pivot.unsqueeze(0).unsqueeze(0)  # (B, S, H, N, D)

    def _decode_weights(self, transformed: torch.Tensor) -> torch.Tensor:
        """Decode transformed coords to weight matrix.
        transformed: (..., H, N, D) -> (..., out, in)
        """
        # Flatten coords per head: (..., H, N*D)
        *batch_shape, H, N, D = transformed.shape
        flat = transformed.reshape(*batch_shape, H, N * D)

        if self.num_heads == 1:
            flat = flat.squeeze(-2)                       # (..., N*D)
            return self.decoder(flat)                     # (..., out, in)
        else:
            # Per-head decode -> stack
            # flat: (..., H, N*D)
            head_weights = self.decoder(flat)             # (..., H, out/H, in)
            # Concatenate heads along out dimension
            if len(batch_shape) == 0:
                return head_weights.reshape(self.out_features, self.in_features)
            else:
                return head_weights.reshape(*batch_shape, self.out_features, self.in_features)

    def generate_weight(self, context: Optional[torch.Tensor] = None):
        """Generate the effective weight matrix.
        context: (B, cond_dim) for seq_conditioned, None for static
        Returns: W_eff (..., out, in) and bias (out,)
        """
        angle = self._get_angle(context)
        transformed = self._transform_coords(angle)
        W_geom = self._decode_weights(transformed)

        if self.mode == "replace":
            W_eff = W_geom
        elif self.mode == "residual":
            if W_geom.dim() == 2:
                W_eff = self.base_weight + self.lam * W_geom
            else:
                W_eff = self.base_weight.unsqueeze(0) + self.lam * W_geom
        else:  # factorized handled by caller
            W_eff = self.lam * W_geom

        return W_eff, self.base_bias

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply the geometric linear layer.
        x: (B, S, in_features)
        context: (B, cond_dim) for seq_conditioned, (B, S, cond_dim) for token_conditioned, None for static
        Returns: (B, S, out_features)
        """
        W_eff, bias = self.generate_weight(context)

        if W_eff.dim() == 2:
            # Static: single weight for all samples -- standard matmul
            return F.linear(x, W_eff, bias)
        elif W_eff.dim() == 3:
            # Seq-conditioned: per-sample weight (B, out, in)
            out = torch.einsum('bsi,boi->bso', x, W_eff)
            return out + bias
        else:
            # Token-conditioned: per-token weight (B, S, out, in)
            out = torch.einsum('bsi,bsoi->bso', x, W_eff)
            return out + bias

    def get_diagnostics(self) -> dict:
        """Return geometry diagnostics for logging."""
        with torch.no_grad():
            diag = {}
            # Rotation magnitudes
            if self.conditioning == "static":
                diag["angle_mag"] = self.angle.abs().mean().item()
            # Scale
            if self.use_scale:
                s = F.softplus(self.scale_logit)
                diag["scale_mean"] = s.mean().item()
                diag["scale_std"] = s.std().item()
            # Lambda
            if hasattr(self, "lam"):
                diag["lambda"] = self.lam.item()
            # Coord spread
            diag["coord_std"] = self.coords.std().item()
            # Pivot drift
            if self.use_pivot_offset:
                diag["pivot_offset_norm"] = self.pivot_offset.norm().item()
            return diag
