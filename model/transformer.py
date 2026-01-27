"""
Transformer Models
==================
Full model architectures including DendriticTransformer.
"""

import torch
import torch.nn as nn
from .components import RMSNorm, StandardMLP, MatryoshkaProj
from .attention import MultiHeadAttention
from ..core.layers import DendriticMLP

class TransformerBlock(nn.Module):
    """Standard Transformer Block."""
    def __init__(self, dim, num_heads, mlp_hidden_dim, dropout, num_kv_heads=None, use_swiglu=True, 
                 max_seq_length=16384, use_checkpointing=False):
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

class DendriticBlock(nn.Module):
    """Mixed Transformer Block supporting Dendritic MLPs."""
    def __init__(self, dim, num_heads, dropout, num_kv_heads=None, use_dendritic=True, 
                 use_swiglu=True, max_seq_length=16384, use_checkpointing=False,
                 # Dendritic Params
                 expand_factor=4,
                 use_dendritic_ffn=False,
                 template_window_size=64,
                 template_tau=1.0,
                 template_branch_count=1,
                 template_use_autotune=True):
        super().__init__()
        
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, num_kv_heads, dropout, max_seq_length=max_seq_length)
        self.norm2 = RMSNorm(dim)
        self.use_checkpointing = use_checkpointing
        
        hidden_features = int(dim * expand_factor)
        
        if use_dendritic:
            self.mlp = DendriticMLP(
                in_features=dim,
                hidden_features=hidden_features,
                dropout_rate=dropout,
                use_dendritic_ffn=use_dendritic_ffn,
                template_window_size=template_window_size,
                template_tau=template_tau,
                template_branch_count=template_branch_count,
                template_use_autotune=template_use_autotune
            )
        else:
            self.mlp = StandardMLP(dim, hidden_features, dropout, use_swiglu=use_swiglu)

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

class DendriticTransformer(nn.Module):
    """
    Transformer with dendritic MLP layers.
    """
    def __init__(self, 
                 vocab_size, 
                 seq_length, 
                 dim, 
                 num_heads, 
                 num_layers, 
                 dropout=0.1, 
                 num_kv_heads=None,
                 backbone=None, 
                 backbone_dim=None,
                 use_checkpointing=False,
                 # Dendritic Configuration
                 use_dendritic=True,
                 dendritic_indices=None, # List of layer indices to use dendritic MLP, None = all
                 use_dendritic_ffn=False,
                 expand_factor=4,
                 # Template Params
                 template_window_size=64, 
                 template_tau=1.0,
                 template_branch_count=1, 
                 template_use_autotune=True):
        super().__init__()
        
        self.dim = dim
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.backbone = backbone

        if backbone is not None and backbone_dim is not None and backbone_dim != dim:
            self.token_embed = None
            self.backbone_proj = MatryoshkaProj(backbone_dim, dim)
        else:
            self.token_embed = nn.Embedding(vocab_size, dim)
            self.backbone_proj = None
            
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList()
        
        # Determine dendritic layers
        if dendritic_indices is None:
            # Check if user wants dendritic usage at all? 
            # If use_dendritic=True and indices=None, use all.
            is_dendritic_layer = [use_dendritic] * num_layers
        else:
            is_dendritic_layer = [i in dendritic_indices for i in range(num_layers)]

        for i in range(num_layers):
            self.blocks.append(DendriticBlock(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                num_kv_heads=num_kv_heads,
                use_dendritic=is_dendritic_layer[i],
                use_swiglu=True, # Baseline fallback
                max_seq_length=seq_length,
                use_checkpointing=use_checkpointing,
                expand_factor=expand_factor,
                use_dendritic_ffn=use_dendritic_ffn,
                template_window_size=template_window_size,
                template_tau=template_tau,
                template_branch_count=template_branch_count,
                template_use_autotune=template_use_autotune
            ))
            
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        # Weight tying
        if self.token_embed is not None:
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
        if self.backbone is not None:
            tok_emb = self.backbone(x)
            if self.backbone_proj is not None:
                tok_emb = self.backbone_proj(tok_emb)
        else:
            tok_emb = self.token_embed(x)

        x = self.dropout(tok_emb)
        
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x)

    def set_template_tau(self, tau):
        """Dynamic temperature adjustment."""
        for block in self.blocks:
            if hasattr(block.mlp, 'dendritic_expand'):
                # Check for template layer
                if hasattr(block.mlp.dendritic_expand, 'tau'):
                    block.mlp.dendritic_expand.tau = tau

    def compute_diversity_loss(self):
        loss = 0.0
        for block in self.blocks:
            if hasattr(block.mlp, 'compute_diversity_loss'):
                loss += block.mlp.compute_diversity_loss()
        return loss
