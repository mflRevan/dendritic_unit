"""
Dendritic Layer Module - Core Implementations
=============================================

This module provides the core dendritic layer implementations:
1.  **DendriticLayerSiLU_Template**: Template-gated MLP using fused Triton kernels (GLU-style).
2.  **DendriticLayerSiLU_FFN**: Pure dendritic projection where templates are the features (FFN-style).
3.  **DendriticMLP**: High-level block wrapper supporting expansion/contraction and dropout.

These layers eliminate explicit activation functions in favor of local competition
(Softmax) and coactivation (NMDA-like) mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch._dynamo

# Suppress Dynamo errors for Triton compatibility
torch._dynamo.config.suppress_errors = True

class DendriticLayerSiLU_Template(nn.Module):
    """
    Template-gated dendritic layer (GLU-style).
    
    Formula: output = Gate(x, Templates) * (x @ Weights.T)
    
    Uses Fused Triton Kernels for efficiency.
    Gate uses local absolute-softmax competition and global coactivation.
    """

    def __init__(
        self,
        in_features,
        out_features,
        window_size=64,
        tau=1.0,
        branch_count=1,
        use_autotune=True,
    ):
        super().__init__()

        if in_features % (branch_count * window_size) != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by branch_count * window_size"
            )
        
        if branch_count < 1:
            raise ValueError(f"branch_count must be at least 1, got {branch_count}")

        self.in_features = in_features
        self.out_features = out_features
        self.window_size = window_size
        self.branch_count = branch_count
        self.windows_per_branch = in_features // (branch_count * window_size)
        self.num_windows = self.branch_count * self.windows_per_branch
        self.tau = tau
        self.use_autotune = use_autotune

        # Linear weights for data path (up projection)
        self.weights = nn.Parameter(torch.empty(out_features, in_features))

        # Template for gate path: [H, num_windows * W]
        self.template_flat = nn.Parameter(
            torch.empty(out_features, self.num_windows * window_size)
        )

        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        nn.init.normal_(self.template_flat, std=0.02)

    def forward(self, x):
        original_shape = x.shape

        if x.dim() == 3:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
            seq_ctx = (B, T)
        else:
            x_flat = x
            seq_ctx = None
        
        N = x_flat.shape[0]
        
        # Adaptive block sizing
        if N <= 256:
            block_n, block_h = 32, 32
        else:
            block_n, block_h = 64, 16
        
        if self.branch_count == 1:
            from .template_gate_fused_up import TemplateGateFusedUpFunction
            out = TemplateGateFusedUpFunction.apply(
                x_flat, self.template_flat, self.weights,
                self.tau, self.window_size, self.num_windows,
                block_n, block_h, self.out_features
            )
        else:
            from .template_gate_fused_up_branched import TemplateGateFusedUpBranchedFunction
            out = TemplateGateFusedUpBranchedFunction.apply(
                x_flat, self.template_flat, self.weights,
                self.tau, self.window_size, self.branch_count, self.windows_per_branch,
                block_n, block_h, self.out_features
            )

        if seq_ctx is not None:
            B, T = seq_ctx
            out = out.reshape(B, T, self.out_features)

        return out

    def compute_diversity_loss(self):
        return torch.tensor(0.0, device=self.weights.device)


class DendriticLayerSiLU_FFN(nn.Module):
    """
    Dendritic FFN layer - Template Direct Projection.
    
    Formula: output = Gate(x, Templates)
             where Gate includes weighted sum of templates.
    
    No separate linear weights matrix. The templates themselves are the weights.
    """

    def __init__(
        self,
        in_features,
        out_features,
        window_size=64,
        tau=1.0,
        branch_count=1,
        use_autotune=True,
    ):
        super().__init__()

        if in_features % (branch_count * window_size) != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by branch_count * window_size"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.window_size = window_size
        self.branch_count = branch_count
        self.windows_per_branch = in_features // (branch_count * window_size)
        self.num_windows = self.branch_count * self.windows_per_branch
        self.tau = tau
        self.use_autotune = use_autotune

        # Templates are the only parameters
        self.template_flat = nn.Parameter(
            torch.empty(out_features, self.num_windows * window_size)
        )

        nn.init.kaiming_normal_(self.template_flat.view(-1, window_size), mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        original_shape = x.shape

        if x.dim() == 3:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
            seq_ctx = (B, T)
        else:
            x_flat = x
            seq_ctx = None
        
        N = x_flat.shape[0]
        
        if N <= 256:
            block_n, block_h = 32, 32
        else:
            block_n, block_h = 64, 16
        
        if self.branch_count == 1:
            from .template_gate_ffn import TemplateGateFFNFunction
            out = TemplateGateFFNFunction.apply(
                x_flat, self.template_flat,
                self.tau, self.window_size, self.num_windows,
                block_n, block_h, self.out_features
            )
        else:
            from .template_gate_ffn_branched import TemplateGateFFNBranchedFunction
            out = TemplateGateFFNBranchedFunction.apply(
                x_flat, self.template_flat,
                self.tau, self.window_size, self.branch_count, self.windows_per_branch,
                block_n, block_h, self.out_features
            )

        if seq_ctx is not None:
            B, T = seq_ctx
            out = out.reshape(B, T, self.out_features)

        return out

    def compute_diversity_loss(self):
        return torch.tensor(0.0, device=self.template_flat.device)


class DendriticMLP(nn.Module):
    """
    Main Dendritic MLP Block.
    
    Architecture:
    Input -> [Expansion Layer] -> Hidden -> [Contraction Layer] -> Output
    
    Modes:
    1. Template-GLU (default)
    2. Template-FFN (use_dendritic_ffn=True)
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        dropout_rate=0.1,
        use_dendritic_ffn=False,
        use_expand=True,
        # Template-specific
        template_window_size=64,
        template_tau=1.0,
        template_branch_count=1,
        template_use_autotune=True,
    ):
        super().__init__()

        self.use_expand = use_expand
        self.use_dendritic_ffn = use_dendritic_ffn
        
        if use_expand:
            if hidden_features is None:
                hidden_features = in_features * 4
            out_features = hidden_features
        else:
            out_features = in_features

        # Initialize the dendritic core layer
        if use_dendritic_ffn:
            self.dendritic_expand = DendriticLayerSiLU_FFN(
                in_features=in_features,
                out_features=out_features,
                window_size=template_window_size,
                tau=template_tau,
                branch_count=template_branch_count,
                use_autotune=template_use_autotune,
            )
        else:
            self.dendritic_expand = DendriticLayerSiLU_Template(
                in_features=in_features,
                out_features=out_features,
                window_size=template_window_size,
                tau=template_tau,
                branch_count=template_branch_count,
                use_autotune=template_use_autotune,
            )

        # Contraction/Down Projection
        if use_expand:
            self.linear_contract = nn.Linear(out_features, in_features)
        else:
            self.linear_contract = None
            
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h = self.dendritic_expand(x)
        if self.linear_contract is not None:
            out = self.linear_contract(h)
        else:
            out = h
        return self.dropout(out)

    def compute_diversity_loss(self):
        return self.dendritic_expand.compute_diversity_loss()
