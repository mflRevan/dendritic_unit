"""
Spinformer: Quaternion-Augmented Transformer
=============================================

The Spinformer introduces quaternion rotations as a geometric inductive bias
into the Transformer residual stream. The residual stream is chunked into 4D
quaternion groups, and each layer applies learned rotations before processing.

Key design choices (configurable):
- rotation_target: "mlp" | "attn" | "both" — which sublayers get rotated input
- rotation_mode: "local" | "global" — local: only the sublayer sees rotated input;
  global: the rotation persists in the residual stream for subsequent layers
"""

import torch
import torch.nn as nn
from .components import RMSNorm, StandardMLP
from .attention import MultiHeadAttention
from .quaternion import QuaternionRotationLayer


class SpinformerBlock(nn.Module):
    """
    Transformer block with quaternion rotation pre-filtering.
    
    The rotation acts as a geometric filter that refines the input
    before the sublayer processes it.
    
    Modes:
    - local: x_rot = rotate(norm(x)); sublayer_out = sublayer(x_rot); x = x + sublayer_out
      (rotation is only seen by the sublayer, residual is unrotated)
    - global: x = rotate(x); x = x + sublayer(norm(x))
      (rotation persists in the stream, compounding across layers)
    - gated: x = x + gate * (rotate(x) - x) where gate = sigmoid(learnable scalar)
      (global rotation with learned interpolation, starts at identity)
    - adaptive: x = x + per_token_gate(x) * (rotate(x) - x)
      (input-dependent gating: each token decides its own rotation strength)
    """
    def __init__(self, dim, num_heads, mlp_hidden_dim, dropout=0.0, num_kv_heads=None,
                 use_swiglu=True, max_seq_length=16384, use_checkpointing=False,
                 rotation_target="both", rotation_mode="local", layer_idx=0):
        super().__init__()
        
        self.rotation_target = rotation_target
        self.rotation_mode = rotation_mode
        self.layer_idx = layer_idx
        
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, num_kv_heads, dropout, max_seq_length=max_seq_length)
        self.norm2 = RMSNorm(dim)
        self.mlp = StandardMLP(dim, mlp_hidden_dim, dropout, use_swiglu=use_swiglu)
        self.use_checkpointing = use_checkpointing
        
        # Quaternion rotation layers
        if rotation_target in ("attn", "both"):
            self.spin_attn = QuaternionRotationLayer(dim)
        else:
            self.spin_attn = None
            
        if rotation_target in ("mlp", "both"):
            self.spin_mlp = QuaternionRotationLayer(dim)
        else:
            self.spin_mlp = None
        
        # Gated mode: learnable scalar gate per sublayer, init to -2 → sigmoid(-2)≈0.12
        if rotation_mode == "gated":
            if self.spin_attn is not None:
                self.gate_attn = nn.Parameter(torch.tensor(-2.0))
            if self.spin_mlp is not None:
                self.gate_mlp = nn.Parameter(torch.tensor(-2.0))
        
        # Adaptive mode: per-token gate via linear projection
        if rotation_mode == "adaptive":
            if self.spin_attn is not None:
                self.adaptive_gate_attn = nn.Sequential(
                    nn.Linear(dim, 1, bias=True),
                    nn.Sigmoid(),
                )
                nn.init.zeros_(self.adaptive_gate_attn[0].weight)
                nn.init.constant_(self.adaptive_gate_attn[0].bias, -2.0)
            if self.spin_mlp is not None:
                self.adaptive_gate_mlp = nn.Sequential(
                    nn.Linear(dim, 1, bias=True),
                    nn.Sigmoid(),
                )
                nn.init.zeros_(self.adaptive_gate_mlp[0].weight)
                nn.init.constant_(self.adaptive_gate_mlp[0].bias, -2.0)

    def _apply_rotation(self, x, spin_layer, mode_variant):
        """Apply rotation with the configured gating strategy."""
        if mode_variant == "local":
            return spin_layer(x)  # caller passes norm(x)
        elif mode_variant == "global":
            return spin_layer(x)
        elif mode_variant == "gated":
            gate_param = self.gate_attn if spin_layer is self.spin_attn else self.gate_mlp
            gate = torch.sigmoid(gate_param)
            x_rot = spin_layer(x)
            return x + gate * (x_rot - x)
        elif mode_variant == "adaptive":
            gate_module = self.adaptive_gate_attn if spin_layer is self.spin_attn else self.adaptive_gate_mlp
            gate = gate_module(x)  # [B, S, 1]
            x_rot = spin_layer(x)
            return x + gate * (x_rot - x)

    def forward(self, x):
        # --- Attention sublayer ---
        if self.spin_attn is not None:
            if self.rotation_mode == "global":
                x = self.spin_attn(x)
                x = x + self.attn(self.norm1(x))
            elif self.rotation_mode in ("gated", "adaptive"):
                x = self._apply_rotation(x, self.spin_attn, self.rotation_mode)
                x = x + self.attn(self.norm1(x))
            else:  # local
                x_normed = self.norm1(x)
                x_rotated = self.spin_attn(x_normed)
                x = x + self.attn(x_rotated)
        else:
            x = x + self.attn(self.norm1(x))
        
        # --- MLP sublayer ---
        if self.spin_mlp is not None:
            if self.rotation_mode == "global":
                x = self.spin_mlp(x)
                x = x + self.mlp(self.norm2(x))
            elif self.rotation_mode in ("gated", "adaptive"):
                x = self._apply_rotation(x, self.spin_mlp, self.rotation_mode)
                x = x + self.mlp(self.norm2(x))
            else:  # local
                x_normed = self.norm2(x)
                x_rotated = self.spin_mlp(x_normed)
                x = x + self.mlp(x_rotated)
        else:
            x = x + self.mlp(self.norm2(x))
        
        return x


class Spinformer(nn.Module):
    """
    Decoder-only Spinformer: Transformer with quaternion rotation inductive bias.
    
    Args:
        rotation_target: "mlp" | "attn" | "both" — which sublayers get rotated
        rotation_mode: "local" | "global" — whether rotation persists in residual
    """
    def __init__(self,
                 vocab_size,
                 seq_length,
                 dim,
                 num_heads,
                 num_layers,
                 dropout=0.0,
                 num_kv_heads=None,
                 expand_factor=4,
                 use_swiglu=True,
                 use_checkpointing=False,
                 rotation_target="both",
                 rotation_mode="local"):
        super().__init__()
        
        assert dim % 4 == 0, f"dim must be divisible by 4 for quaternion chunking, got {dim}"
        
        self.dim = dim
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.rotation_target = rotation_target
        self.rotation_mode = rotation_mode
        
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)
        
        mlp_hidden_dim = int(dim * expand_factor)
        
        self.blocks = nn.ModuleList([
            SpinformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                dropout=dropout,
                num_kv_heads=num_kv_heads,
                use_swiglu=use_swiglu,
                max_seq_length=seq_length,
                use_checkpointing=use_checkpointing,
                rotation_target=rotation_target,
                rotation_mode=rotation_mode,
                layer_idx=i,
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
    
    def forward(self, x):
        x = self.dropout(self.token_embed(x))
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return self.head(x)
    
    def get_spin_stats(self):
        """Get statistics about learned rotation axes, angles, and gates for analysis."""
        stats = {}
        for i, block in enumerate(self.blocks):
            layer_stats = {}
            if block.spin_attn is not None:
                axes = block.spin_attn.get_unit_axes()
                layer_stats['attn_axes'] = axes.detach()
                layer_stats['attn_angle_weight_norm'] = block.spin_attn.angle_proj.weight.norm().item()
            if block.spin_mlp is not None:
                axes = block.spin_mlp.get_unit_axes()
                layer_stats['mlp_axes'] = axes.detach()
                layer_stats['mlp_angle_weight_norm'] = block.spin_mlp.angle_proj.weight.norm().item()
            # Report gate values for gated mode
            if block.rotation_mode == "gated":
                if hasattr(block, 'gate_attn'):
                    layer_stats['gate_attn'] = torch.sigmoid(block.gate_attn).item()
                if hasattr(block, 'gate_mlp'):
                    layer_stats['gate_mlp'] = torch.sigmoid(block.gate_mlp).item()
            stats[f'layer_{i}'] = layer_stats
        return stats
