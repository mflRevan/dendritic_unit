"""
Transformer Model
=================
Clean, modular Transformer architecture with RMSNorm, SwiGLU, GQA, and RoPE.
"""

import torch
import torch.nn as nn
from .components import RMSNorm, StandardMLP
from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: RMSNorm -> Attention -> RMSNorm -> MLP."""
    def __init__(self, dim, num_heads, mlp_hidden_dim, dropout=0.0, num_kv_heads=None, 
                 use_swiglu=True, max_seq_length=16384, use_checkpointing=False):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, num_kv_heads, dropout, max_seq_length=max_seq_length)
        self.norm2 = RMSNorm(dim)
        self.mlp = StandardMLP(dim, mlp_hidden_dim, dropout, use_swiglu=use_swiglu)
        self.use_checkpointing = use_checkpointing

    def _attn_residual(self, x):
        return self.attn(self.norm1(x))
    
    def _mlp_residual(self, x):
        return self.mlp(self.norm2(x))

    def forward(self, x):
        if self.use_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            x = x + checkpoint(self._attn_residual, x, use_reentrant=False)
            x = x + checkpoint(self._mlp_residual, x, use_reentrant=False)
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """
    Decoder-only Transformer for sequence modeling.
    
    Features:
    - RMSNorm (pre-norm architecture)
    - SwiGLU MLP
    - Grouped Query Attention (GQA) with RoPE
    - Weight tying (embedding <-> output head)
    - Gradient checkpointing support
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
                 use_checkpointing=False):
        super().__init__()
        
        self.dim = dim
        self.seq_length = seq_length
        self.num_layers = num_layers

        self.token_embed = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)
        
        mlp_hidden_dim = int(dim * expand_factor)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                dropout=dropout,
                num_kv_heads=num_kv_heads,
                use_swiglu=use_swiglu,
                max_seq_length=seq_length,
                use_checkpointing=use_checkpointing,
            )
            for _ in range(num_layers)
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
