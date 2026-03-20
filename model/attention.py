"""
Attention Modules
=================
Multi-Head Attention with GQA and RoPE support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import RoPE


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Grouped Query Attention (GQA) and RoPE.
    Uses PyTorch's scaled_dot_product_attention (Flash Attention when available).
    """
    def __init__(self, dim, num_heads, num_kv_heads=None, dropout=0.0, max_seq_length=16384):
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

        q_size = self.num_heads * self.head_dim
        kv_size = 2 * self.num_kv_heads * self.head_dim
        total_qkv_dim = q_size + kv_size

        self.qkv_proj = nn.Linear(dim, total_qkv_dim, bias=False)
        self.proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = dropout
        self.rope = RoPE(self.head_dim, max_seq_length=max_seq_length)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv_proj(x)
        
        q_size = self.num_heads * self.head_dim
        k_size = self.num_kv_heads * self.head_dim
        
        q = qkv[..., :q_size]
        k = qkv[..., q_size:q_size + k_size]
        v = qkv[..., q_size + k_size:]
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        q = self.rope(q)
        k = self.rope(k)
        
        # GQA: expand KV heads to match Q heads (zero-copy expand)
        if self.num_queries_per_kv > 1:
            k = k[:, :, None, :, :].expand(-1, -1, self.num_queries_per_kv, -1, -1).reshape(batch_size, self.num_heads, seq_len, self.head_dim)
            v = v[:, :, None, :, :].expand(-1, -1, self.num_queries_per_kv, -1, -1).reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )

        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        out = self.proj(out)
        out = self.dropout(out)

        return out
