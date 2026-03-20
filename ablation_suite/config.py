"""
Configuration classes for the ablation suite.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class TaskConfig:
    """Configuration for a specific task."""
    name: str
    train_seq_len: int = 32
    val_seq_len: int = 32
    test_seq_len: int = 32
    train_samples: int = 20000
    val_samples: int = 2000
    test_samples: int = 2000
    max_seq_len: int = 512
    vocab_size: int = 256
    
    # Task-specific params
    modulo: int = 97        # For modular arithmetic
    num_bits: int = 32      # For bitwise add


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    name: str
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512       # 4x d_model
    dropout: float = 0.0
    use_swiglu: bool = True
    num_kv_heads: Optional[int] = None  # None = MHA, set for GQA
    
    # Spinformer params
    arch: str = "transformer"  # "transformer" | "spinformer" | "geofield"
    rotation_target: str = "both"  # "mlp" | "attn" | "both"
    rotation_mode: str = "local"   # "local" | "global"
    
    # GeoField params
    geo_target: str = "attn_out"         # "attn_out"|"value"|"both_vo"|"mlp_up"|"mlp_down"|"block_residual"
    geo_mode: str = "residual"           # "replace"|"residual"|"factorized"
    geo_conditioning: str = "static"     # "static"|"seq_conditioned"
    geo_num_coords: int = 32
    geo_rank: int = 0                    # 0=full decode, >0=low-rank
    geo_use_scale: bool = False
    geo_use_pivot_offset: bool = False
    geo_num_heads: int = 1               # 1=shared, >1=per-head geo transforms
    geo_lam_init: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training."""
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    use_compile: bool = True
    
    # Per-task overrides: task_name -> {field: value}
    task_overrides: Dict[str, Dict[str, Any]] = None
    
    def for_task(self, task_name: str) -> 'TrainingConfig':
        """Return a config with task-specific overrides applied."""
        if not self.task_overrides or task_name not in self.task_overrides:
            return self
        overrides = self.task_overrides[task_name]
        kwargs = {
            'num_epochs': overrides.get('num_epochs', self.num_epochs),
            'batch_size': overrides.get('batch_size', self.batch_size),
            'learning_rate': overrides.get('learning_rate', self.learning_rate),
            'weight_decay': overrides.get('weight_decay', self.weight_decay),
            'grad_clip': overrides.get('grad_clip', self.grad_clip),
            'use_compile': overrides.get('use_compile', self.use_compile),
        }
        return TrainingConfig(**kwargs)


# Pre-defined model configurations
MODEL_CONFIGS = {
    "baseline": ModelConfig(
        name="Baseline (SwiGLU)",
    ),
    "spin_local_both": ModelConfig(
        name="Spinformer (local, attn+mlp)",
        arch="spinformer",
        rotation_target="both",
        rotation_mode="local",
    ),
    "spin_local_mlp": ModelConfig(
        name="Spinformer (local, mlp-only)",
        arch="spinformer",
        rotation_target="mlp",
        rotation_mode="local",
    ),
    "spin_global_both": ModelConfig(
        name="Spinformer (global, attn+mlp)",
        arch="spinformer",
        rotation_target="both",
        rotation_mode="global",
    ),
    "spin_global_mlp": ModelConfig(
        name="Spinformer (global, mlp-only)",
        arch="spinformer",
        rotation_target="mlp",
        rotation_mode="global",
    ),
    "spin_gated_mlp": ModelConfig(
        name="Spinformer (gated, mlp-only)",
        arch="spinformer",
        rotation_target="mlp",
        rotation_mode="gated",
    ),
    "spin_adaptive_mlp": ModelConfig(
        name="Spinformer (adaptive, mlp-only)",
        arch="spinformer",
        rotation_target="mlp",
        rotation_mode="adaptive",
    ),

    # ---- GeoField models (Phase 1: attn_out) ----
    "geo_attn_out_static": ModelConfig(
        name="GeoField (attn_out, static, residual)",
        arch="geofield",
        geo_target="attn_out",
        geo_mode="residual",
        geo_conditioning="static",
    ),
    "geo_attn_out_cond": ModelConfig(
        name="GeoField (attn_out, seq_cond, residual)",
        arch="geofield",
        geo_target="attn_out",
        geo_mode="residual",
        geo_conditioning="seq_conditioned",
    ),
    "geo_attn_out_replace": ModelConfig(
        name="GeoField (attn_out, static, replace)",
        arch="geofield",
        geo_target="attn_out",
        geo_mode="replace",
        geo_conditioning="static",
    ),
    "geo_attn_out_perhead": ModelConfig(
        name="GeoField (attn_out, static, per-head)",
        arch="geofield",
        geo_target="attn_out",
        geo_mode="residual",
        geo_conditioning="static",
        geo_num_heads=4,
    ),
    "geo_attn_out_scale": ModelConfig(
        name="GeoField (attn_out, static, +scale)",
        arch="geofield",
        geo_target="attn_out",
        geo_mode="residual",
        geo_conditioning="static",
        geo_use_scale=True,
    ),
    "geo_attn_out_pivot": ModelConfig(
        name="GeoField (attn_out, static, +pivot)",
        arch="geofield",
        geo_target="attn_out",
        geo_mode="residual",
        geo_conditioning="static",
        geo_use_pivot_offset=True,
    ),
    "geo_attn_out_full": ModelConfig(
        name="GeoField (attn_out, static, rot+scale+pivot)",
        arch="geofield",
        geo_target="attn_out",
        geo_mode="residual",
        geo_conditioning="static",
        geo_use_scale=True,
        geo_use_pivot_offset=True,
    ),
    "geo_attn_out_lowrank": ModelConfig(
        name="GeoField (attn_out, static, low-rank 16)",
        arch="geofield",
        geo_target="attn_out",
        geo_mode="residual",
        geo_conditioning="static",
        geo_rank=16,
    ),
    # ---- Phase 2: other insertion points ----
    "geo_value_static": ModelConfig(
        name="GeoField (value, static, residual)",
        arch="geofield",
        geo_target="value",
        geo_mode="residual",
        geo_conditioning="static",
    ),
    "geo_both_vo_static": ModelConfig(
        name="GeoField (V+O, static, residual)",
        arch="geofield",
        geo_target="both_vo",
        geo_mode="residual",
        geo_conditioning="static",
    ),
    "geo_mlp_down_static": ModelConfig(
        name="GeoField (mlp_down, static, residual)",
        arch="geofield",
        geo_target="mlp_down",
        geo_mode="residual",
        geo_conditioning="static",
    ),
    "geo_mlp_up_static": ModelConfig(
        name="GeoField (mlp_up, static, residual)",
        arch="geofield",
        geo_target="mlp_up",
        geo_mode="residual",
        geo_conditioning="static",
    ),
    "geo_block_residual": ModelConfig(
        name="GeoField (block_residual, static)",
        arch="geofield",
        geo_target="block_residual",
        geo_mode="residual",  # not used for block_residual, but kept for compat
        geo_conditioning="static",
    ),
}


# Pre-defined task configurations
TASK_CONFIGS = {
    "sorting": TaskConfig(
        name="sorting",
        train_seq_len=32,
        val_seq_len=32,
        test_seq_len=32,
        max_seq_len=512,
        vocab_size=256,
        train_samples=20000,
        val_samples=2000,
        test_samples=2000,
    ),
    
    "modular_arith": TaskConfig(
        name="modular_arith",
        train_seq_len=16,
        val_seq_len=16,
        test_seq_len=16,
        max_seq_len=512,
        vocab_size=128,
        modulo=97,
        train_samples=20000,
        val_samples=2000,
        test_samples=2000,
    ),
    
    "reversal": TaskConfig(
        name="reversal",
        train_seq_len=32,
        val_seq_len=32,
        test_seq_len=32,
        max_seq_len=512,
        vocab_size=256,
        train_samples=20000,
        val_samples=2000,
        test_samples=2000,
    ),
    
    "bitwise_add": TaskConfig(
        name="bitwise_add",
        train_seq_len=16,
        val_seq_len=16,
        test_seq_len=16,
        max_seq_len=512,
        vocab_size=4,
        num_bits=16,
        train_samples=20000,
        val_samples=2000,
        test_samples=2000,
    ),
    
    "parity": TaskConfig(
        name="parity",
        train_seq_len=32,
        val_seq_len=32,
        test_seq_len=32,
        max_seq_len=1024,
        vocab_size=4,
        train_samples=10000,
        val_samples=1000,
        test_samples=2000,
    ),
}


def get_model_configs() -> Dict[str, ModelConfig]:
    """Get all model configurations."""
    return MODEL_CONFIGS


def get_task_configs() -> Dict[str, TaskConfig]:
    """Get configurations for all tasks."""
    return TASK_CONFIGS
