"""
Model Components
================
Common building blocks: RMSNorm, RoPE, SwiGLU MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class RoPE(nn.Module):
    """Rotary Position Embeddings (RoPE)."""
    def __init__(self, dim: int, base: int = 10_000, max_seq_length: int = 8192):
        super().__init__()
        self.base = base
        self.dim = dim
        self.max_seq_length = max_seq_length
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        t = torch.arange(max_seq_length).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)
    
    def _neg_half(self, x: torch.Tensor) -> torch.Tensor:
        d_2 = self.dim // 2
        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        
        if seq_len > self.max_seq_length:
            t = torch.arange(seq_len).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos()[None, None, :, :]
            sin = emb.sin()[None, None, :, :]
        else:
            cos = self.cos_cached
            sin = self.sin_cached
        
        if x.dim() == 4:
            cos = cos[:, :, :seq_len, :].to(device=x.device, dtype=x.dtype)
            sin = sin[:, :, :seq_len, :].to(device=x.device, dtype=x.dtype)
        else:
            cos = cos[0, 0, :seq_len, :].to(device=x.device, dtype=x.dtype)
            sin = sin[0, 0, :seq_len, :].to(device=x.device, dtype=x.dtype)
        
        return (x * cos) + (self._neg_half(x) * sin)


class StandardMLP(nn.Module):
    """Feed-forward block with optional SwiGLU activation."""
    def __init__(self, dim, hidden_dim, dropout=0.0, use_swiglu=True):
        super().__init__()
        self.use_swiglu = use_swiglu

        if self.use_swiglu:
            self.w_fused = nn.Linear(dim, 2 * hidden_dim, bias=False)
            self.w_down = nn.Linear(hidden_dim, dim, bias=False)
            self.dropout = nn.Dropout(dropout)
        else:
            self.fc1 = nn.Linear(dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, dim)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.use_swiglu:
            fused = self.w_fused(x)
            gate, data = fused.chunk(2, dim=-1)
            gate = F.silu(gate)
            gated_x = gate * data
            gated_x = self.dropout(gated_x)
            return self.w_down(gated_x)
        else:
            x = self.fc1(x)
            x = F.silu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            return x
