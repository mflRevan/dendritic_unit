"""
GeoField Transformer
====================
Transformer with Geometric Weight-Field Modulation at configurable insertion
points.  Any subset of {Q, K, V, O} attention projections can be replaced
by a GeometricWeightField that generates weights from rotated/scaled 3D
latent coordinates.

Target format: string of characters from {q, k, v, o}
  e.g., "o" (output only), "vo" (value+output), "qkvo" (all four)
Legacy: "attn_out" -> "o", "value" -> "v", "both_vo" -> "vo"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import RMSNorm, StandardMLP
from .attention import MultiHeadAttention
from .geometric_field import GeometricWeightField


# ---------- conditioning source helpers ----------

class AttnPool(nn.Module):
    """Learned attention pooling: weighted sum over sequence positions."""
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1, bias=False)
        nn.init.normal_(self.attn.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D) → (B, D)
        weights = F.softmax(self.attn(x).squeeze(-1), dim=-1)  # (B, S)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


def _compute_context(x: torch.Tensor, source: str, attn_pool=None):
    """Compute conditioning context from hidden states.
    x: (B, S, D)
    Returns: (B, D) for per-seq sources, (B, S, D) for per-token
    """
    if source == "mean_pool":
        return x.mean(dim=1)
    elif source == "last_token":
        return x[:, -1, :]
    elif source == "first_token":
        return x[:, 0, :]
    elif source == "max_pool":
        return x.max(dim=1).values
    elif source == "attn_pool":
        assert attn_pool is not None
        return attn_pool(x)
    elif source == "detached_mean":
        return x.mean(dim=1).detach()
    elif source == "per_token":
        return x  # (B, S, D) — each token is its own context
    else:
        raise ValueError(f"Unknown conditioning source: {source}")


# ---------- target helpers ----------

_LEGACY_MAP = {"attn_out": "o", "value": "v", "both_vo": "vo"}
_ATTN_CHARS = frozenset("qkvo")


def _normalize_target(geo_target: str) -> str:
    """Map legacy target names to new QKVO format."""
    return _LEGACY_MAP.get(geo_target, geo_target).lower()


def _is_attn_target(geo_target: str) -> bool:
    """True if target involves any attention projection."""
    return bool(set(_normalize_target(geo_target)) & _ATTN_CHARS)


# ---------- Attention ----------

class GeoFieldAttention(nn.Module):
    """
    Multi-Head Attention where any subset of {Q, K, V, O} projections can be
    replaced/modulated by a GeometricWeightField.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads=None,
        dropout: float = 0.0,
        max_seq_length: int = 16384,
        # --- geo field config ---
        geo_target: str = "o",
        geo_mode: str = "replace",
        geo_conditioning: str = "static",
        geo_num_coords: int = 32,
        geo_rank: int = 0,
        geo_use_scale: bool = True,
        geo_use_pivot_offset: bool = False,
        geo_num_heads: int = 1,
        geo_lam_init: float = 0.1,
        geo_cond_source: str = "mean_pool",
        geo_shared_controller: bool = False,
        geo_rotation: str = "quaternion",
        geo_coord_dim: int = 3,
    ):
        super().__init__()

        if num_kv_heads is None:
            num_kv_heads = num_heads

        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.geo_conditioning = geo_conditioning
        self.geo_cond_source = geo_cond_source

        # Attention pooling module (if needed)
        self._attn_pool = AttnPool(dim) if geo_cond_source == "attn_pool" else None

        # Parse which projections are geo-modulated
        target = _normalize_target(geo_target)
        self.has_geo_q = 'q' in target
        self.has_geo_k = 'k' in target
        self.has_geo_v = 'v' in target
        self.has_geo_o = 'o' in target

        q_out = num_heads * self.head_dim
        kv_out = num_kv_heads * self.head_dim

        # Determine conditioning mode for GeometricWeightField
        gwf_conditioning = geo_conditioning
        if geo_conditioning == "seq_conditioned" and geo_cond_source == "per_token":
            gwf_conditioning = "token_conditioned"

        geo_kwargs = dict(
            num_coords=geo_num_coords, coord_dim=geo_coord_dim,
            mode=geo_mode, conditioning=gwf_conditioning,
            use_scale=geo_use_scale, use_pivot_offset=geo_use_pivot_offset,
            rank=geo_rank, lam_init=geo_lam_init,
            num_heads=geo_num_heads, cond_dim=dim,
            rotation_type=geo_rotation,
        )

        # Shared controller: one field generates all geo projections
        geo_active = [self.has_geo_q, self.has_geo_k, self.has_geo_v, self.has_geo_o]
        if geo_shared_controller and sum(geo_active) > 1:
            self._shared_field = GeometricWeightField(
                out_features=q_out, in_features=dim, **geo_kwargs)
            self._geo_shared = True
        else:
            self._shared_field = None
            self._geo_shared = False

        # --- Q ---
        if self.has_geo_q:
            self.q_field = self._shared_field if self._geo_shared else \
                GeometricWeightField(out_features=q_out, in_features=dim, **geo_kwargs)
        else:
            self.q_proj = nn.Linear(dim, q_out, bias=False)

        # --- K ---
        if self.has_geo_k:
            self.k_field = self._shared_field if self._geo_shared else \
                GeometricWeightField(out_features=kv_out, in_features=dim, **geo_kwargs)
        else:
            self.k_proj = nn.Linear(dim, kv_out, bias=False)

        # --- V ---
        if self.has_geo_v:
            self.v_field = self._shared_field if self._geo_shared else \
                GeometricWeightField(out_features=kv_out, in_features=dim, **geo_kwargs)
        else:
            self.v_proj = nn.Linear(dim, kv_out, bias=False)

        # --- O ---
        if self.has_geo_o:
            self.o_field = self._shared_field if self._geo_shared else \
                GeometricWeightField(out_features=dim, in_features=dim, **geo_kwargs)
        else:
            self.o_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = dropout

        from .components import RoPE
        self.rope = RoPE(self.head_dim, max_seq_length=max_seq_length)

    def _get_context(self, x: torch.Tensor):
        if self.geo_conditioning == "static":
            return None
        return _compute_context(x, self.geo_cond_source, self._attn_pool)

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        B, S, _ = x.shape
        ctx = context if context is not None else self._get_context(x)

        # Q, K, V projections
        q = self.q_field(x, ctx) if self.has_geo_q else self.q_proj(x)
        k = self.k_field(x, ctx) if self.has_geo_k else self.k_proj(x)
        v = self.v_field(x, ctx) if self.has_geo_v else self.v_proj(x)

        # Reshape to multi-head
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        q = self.rope(q)
        k = self.rope(k)

        # GQA: expand KV heads (zero-copy)
        if self.num_queries_per_kv > 1:
            k = k[:, :, None, :, :].expand(-1, -1, self.num_queries_per_kv, -1, -1).reshape(B, self.num_heads, S, self.head_dim)
            v = v[:, :, None, :, :].expand(-1, -1, self.num_queries_per_kv, -1, -1).reshape(B, self.num_heads, S, self.head_dim)

        # Attention
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )

        out = out.transpose(1, 2).reshape(B, S, self.dim)

        # O projection
        out = self.o_field(out, ctx) if self.has_geo_o else self.o_proj(out)
        out = self.dropout(out)
        return out


# ---------- MLP ----------

class GeoFieldMLP(nn.Module):
    """SwiGLU MLP where up or down projection is geo-modulated."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        geo_target: str = "mlp_down",
        geo_mode: str = "residual",
        geo_conditioning: str = "static",
        geo_num_coords: int = 32,
        geo_rank: int = 0,
        geo_use_scale: bool = False,
        geo_use_pivot_offset: bool = False,
        geo_num_heads: int = 1,
        geo_lam_init: float = 0.1,
        geo_rotation: str = "quaternion",
        geo_coord_dim: int = 3,
        **kwargs,  # absorb extra geo_kwargs (geo_cond_source, geo_shared_controller)
    ):
        super().__init__()
        self.geo_target = geo_target
        self.geo_conditioning = geo_conditioning

        if geo_target == "mlp_up":
            self.geo_up = GeometricWeightField(
                out_features=2 * hidden_dim, in_features=dim,
                num_coords=geo_num_coords, coord_dim=geo_coord_dim,
                mode=geo_mode, conditioning=geo_conditioning,
                use_scale=geo_use_scale, use_pivot_offset=geo_use_pivot_offset,
                rank=geo_rank, lam_init=geo_lam_init,
                num_heads=geo_num_heads, cond_dim=dim,
                rotation_type=geo_rotation,
            )
            self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        elif geo_target == "mlp_down":
            self.w_fused = nn.Linear(dim, 2 * hidden_dim, bias=False)
            self.geo_down = GeometricWeightField(
                out_features=dim, in_features=hidden_dim,
                num_coords=geo_num_coords, coord_dim=geo_coord_dim,
                mode=geo_mode, conditioning=geo_conditioning,
                use_scale=geo_use_scale, use_pivot_offset=geo_use_pivot_offset,
                rank=geo_rank, lam_init=geo_lam_init,
                num_heads=geo_num_heads, cond_dim=dim,
                rotation_type=geo_rotation,
            )

        self.dropout = nn.Dropout(dropout)

    def _get_context(self, x: torch.Tensor):
        if self.geo_conditioning == "static":
            return None
        return x.mean(dim=1)

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        ctx = context if context is not None else self._get_context(x)

        if self.geo_target == "mlp_up":
            fused = self.geo_up(x, ctx)
            gate, data = fused.chunk(2, dim=-1)
            return self.w_down(self.dropout(F.silu(gate) * data))
        else:
            fused = self.w_fused(x)
            gate, data = fused.chunk(2, dim=-1)
            h = self.dropout(F.silu(gate) * data)
            return self.geo_down(h, ctx)


# ---------- Block ----------

class GeoFieldBlock(nn.Module):
    """Pre-norm Transformer block with geometric weight-field modulation."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float = 0.0,
        num_kv_heads=None,
        max_seq_length: int = 16384,
        # --- geo field config ---
        geo_target: str = "o",
        geo_mode: str = "replace",
        geo_conditioning: str = "static",
        geo_num_coords: int = 32,
        geo_rank: int = 0,
        geo_use_scale: bool = True,
        geo_use_pivot_offset: bool = False,
        geo_num_heads: int = 1,
        geo_lam_init: float = 0.1,
        geo_cond_source: str = "mean_pool",
        geo_shared_controller: bool = False,
        geo_rotation: str = "quaternion",
        geo_coord_dim: int = 3,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        mlp_targets = {"mlp_up", "mlp_down"}
        block_target = {"block_residual"}

        geo_kwargs = dict(
            geo_mode=geo_mode, geo_conditioning=geo_conditioning,
            geo_num_coords=geo_num_coords, geo_rank=geo_rank,
            geo_use_scale=geo_use_scale, geo_use_pivot_offset=geo_use_pivot_offset,
            geo_num_heads=geo_num_heads, geo_lam_init=geo_lam_init,
            geo_cond_source=geo_cond_source,
            geo_shared_controller=geo_shared_controller,
            geo_rotation=geo_rotation,
            geo_coord_dim=geo_coord_dim,
        )

        if _is_attn_target(geo_target):
            self.attn = GeoFieldAttention(
                dim, num_heads, num_kv_heads, dropout, max_seq_length,
                geo_target=geo_target, **geo_kwargs,
            )
            self.mlp = StandardMLP(dim, mlp_hidden_dim, dropout, use_swiglu=True)
        elif geo_target in mlp_targets:
            self.attn = MultiHeadAttention(dim, num_heads, num_kv_heads, dropout, max_seq_length)
            self.mlp = GeoFieldMLP(
                dim, mlp_hidden_dim, dropout, geo_target=geo_target, **geo_kwargs,
            )
        elif geo_target in block_target:
            self.attn = MultiHeadAttention(dim, num_heads, num_kv_heads, dropout, max_seq_length)
            self.mlp = StandardMLP(dim, mlp_hidden_dim, dropout, use_swiglu=True)
            self.geo_residual = GeometricWeightField(
                out_features=dim, in_features=dim,
                num_coords=geo_num_coords, coord_dim=geo_coord_dim,
                mode="replace", conditioning=geo_conditioning,
                use_scale=geo_use_scale, use_pivot_offset=geo_use_pivot_offset,
                rank=geo_rank, lam_init=geo_lam_init,
                num_heads=geo_num_heads, cond_dim=dim,
                rotation_type=geo_rotation,
            )
            self.geo_residual_lam = nn.Parameter(torch.tensor(geo_lam_init))
        else:
            raise ValueError(f"Unknown geo_target: {geo_target}")

        self.geo_target = geo_target

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        if hasattr(self.attn, '_get_context'):
            x = x + self.attn(self.norm1(x), context=context)
        else:
            x = x + self.attn(self.norm1(x))

        if hasattr(self.mlp, '_get_context'):
            x = x + self.mlp(self.norm2(x), context=context)
        else:
            x = x + self.mlp(self.norm2(x))

        if self.geo_target == "block_residual":
            ctx = context if context is not None else (
                x.mean(dim=1) if hasattr(self, 'geo_residual') and
                self.geo_residual.conditioning != "static" else None)
            x = x + self.geo_residual_lam * self.geo_residual(x, ctx)
        return x


# ---------- Full Transformer ----------

class GeoFieldTransformer(nn.Module):
    """
    Decoder-only Transformer with Geometric Weight-Field Modulation.
    Drop-in replacement for Transformer, identical API for training.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_length: int,
        dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
        num_kv_heads=None,
        expand_factor: int = 4,
        use_swiglu: bool = True,
        # --- geo config ---
        geo_target: str = "o",
        geo_mode: str = "replace",
        geo_conditioning: str = "static",
        geo_num_coords: int = 32,
        geo_rank: int = 0,
        geo_use_scale: bool = True,
        geo_use_pivot_offset: bool = False,
        geo_num_heads: int = 1,
        geo_lam_init: float = 0.1,
        geo_cond_source: str = "mean_pool",
        geo_shared_controller: bool = False,
        geo_controller_type: str = "local",  # local, ema, gru, first_only
        geo_rotation: str = "quaternion",
        geo_coord_dim: int = 3,
        geo_conditioned_layers: str = "all",  # "all", or comma-separated indices e.g. "0,1"
    ):
        super().__init__()
        self.dim = dim
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.geo_controller_type = geo_controller_type
        self.geo_conditioning = geo_conditioning

        # Parse which layers get conditioning
        if geo_conditioned_layers == "all":
            self._cond_layer_set = set(range(num_layers))
        else:
            self._cond_layer_set = {int(x) for x in geo_conditioned_layers.split(",")}

        self.token_embed = nn.Embedding(vocab_size, dim)
        self.dropout_layer = nn.Dropout(dropout)

        mlp_hidden_dim = int(dim * expand_factor)

        # Cross-layer controller modules
        if geo_controller_type == "gru" and geo_conditioning != "static":
            state_dim = dim // 2
            self.ctrl_gru = nn.GRUCell(dim, state_dim)
            self.ctrl_proj = nn.Linear(state_dim, dim)
            self._ctrl_state_dim = state_dim
        elif geo_controller_type == "ema" and geo_conditioning != "static":
            self.ctrl_alpha = nn.Parameter(torch.tensor(0.5))  # learned EMA decay

        mlp_hidden_dim = int(dim * expand_factor)

        self.blocks = nn.ModuleList([
            GeoFieldBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                dropout=dropout,
                num_kv_heads=num_kv_heads,
                max_seq_length=seq_length,
                geo_target=geo_target,
                geo_mode=geo_mode,
                geo_conditioning=geo_conditioning if i in self._cond_layer_set else "static",
                geo_num_coords=geo_num_coords,
                geo_rank=geo_rank,
                geo_use_scale=geo_use_scale,
                geo_use_pivot_offset=geo_use_pivot_offset,
                geo_num_heads=geo_num_heads,
                geo_lam_init=geo_lam_init,
                geo_cond_source=geo_cond_source,
                geo_shared_controller=geo_shared_controller,
                geo_rotation=geo_rotation,
                geo_coord_dim=geo_coord_dim,
            )
            for i in range(num_layers)
        ])

        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout_layer(self.token_embed(x))

        if self.geo_controller_type == "local" or self.geo_conditioning == "static":
            # Default: each block computes its own context
            for block in self.blocks:
                x = block(x)
        elif self.geo_controller_type == "first_only":
            # Compute context once from input embeddings, reuse across layers
            ctx = x.mean(dim=1)
            for block in self.blocks:
                x = block(x, context=ctx)
        elif self.geo_controller_type == "ema":
            # Exponential moving average across layers
            alpha = torch.sigmoid(self.ctrl_alpha)
            h = torch.zeros_like(x.mean(dim=1))
            for block in self.blocks:
                summary = x.mean(dim=1)
                h = alpha * h + (1 - alpha) * summary
                x = block(x, context=h)
        elif self.geo_controller_type == "gru":
            # GRU state accumulation across layers
            B = x.shape[0]
            h = torch.zeros(B, self._ctrl_state_dim, device=x.device, dtype=x.dtype)
            for block in self.blocks:
                summary = x.mean(dim=1)
                h = self.ctrl_gru(summary, h)
                ctx = self.ctrl_proj(h)
                x = block(x, context=ctx)

        x = self.norm(x)
        return self.head(x)

    def get_geo_stats(self) -> dict:
        """Collect diagnostic statistics from all geometric modules."""
        stats = {}
        for i, block in enumerate(self.blocks):
            layer_stats = {}
            # New QKVO fields
            for proj_name in ('q_field', 'k_field', 'v_field', 'o_field'):
                if hasattr(block.attn, proj_name):
                    prefix = proj_name[0]
                    field = getattr(block.attn, proj_name)
                    for k, v in field.get_diagnostics().items():
                        layer_stats[f"{prefix}_{k}"] = v
            # MLP fields
            if hasattr(block, 'mlp') and hasattr(block.mlp, 'geo_down'):
                for k, v in block.mlp.geo_down.get_diagnostics().items():
                    layer_stats[f"mlp_down_{k}"] = v
            if hasattr(block, 'mlp') and hasattr(block.mlp, 'geo_up'):
                for k, v in block.mlp.geo_up.get_diagnostics().items():
                    layer_stats[f"mlp_up_{k}"] = v
            # Block residual
            if hasattr(block, 'geo_residual'):
                for k, v in block.geo_residual.get_diagnostics().items():
                    layer_stats[f"block_res_{k}"] = v
                layer_stats["block_res_lam"] = block.geo_residual_lam.item()
            if layer_stats:
                stats[f"layer_{i}"] = layer_stats
        return stats